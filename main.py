import argparse
import os
import pickle
from itertools import chain
from typing import Optional, Union

import jsonlines
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.file_utils import PaddingStrategy

from hyper import *
import torch
import transformers
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForMultipleChoice, DataCollatorForSeq2Seq, \
    AutoModelForSequenceClassification, DataCollatorForTokenClassification, DataCollatorWithPadding, \
    PreTrainedTokenizerBase
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import label_ranking_average_precision_score as mrr_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset):
        self.input_ids = tokenized_dataset['input_ids']
        self.labels = tokenized_dataset['labels']
        self.type_ids = tokenized_dataset['type_ids']

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx],
                'labels': self.labels[idx],
                'token_type_ids': self.type_ids[idx]}

    def __len__(self):
        return len(self.labels)


class MyCollator(DataCollatorWithPadding):
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def load_data(tokenizer):
    # train_data = {'input_ids': [], 'type_ids': [], 'labels': []}
    # with jsonlines.open("Data/train.jsonl", 'r') as f:
    #     for line in f:
    #         choice_list = [choice + "[DESC]" + ' '.join(line['desc'][i]) for i, choice in enumerate(line['choice'])]
    #         tokenized_input = tokenizer(text=[line['riddle']] * 5,  text_pair=choice_list,
    #                                     padding=True, truncation=True, max_length=256)
    #         train_data['input_ids'].append(tokenized_input['input_ids'])
    #         train_data['type_ids'].append(tokenized_input['token_type_ids'])
    #         train_data['labels'].append(line['label'])
    #
    # valid_data = {'input_ids': [], 'type_ids': [], 'labels': []}
    # with jsonlines.open("Data/valid.jsonl", 'r') as f:
    #     for line in f:
    #         choice_list = [choice + "[DESC]" + ' '.join(line['desc'][i]) for i, choice in enumerate(line['choice'])]
    #         tokenized_input = tokenizer(text=[line['riddle']] * 5, text_pair=choice_list,
    #                                     padding=True, truncation=True, max_length=256)
    #         valid_data['input_ids'].append(tokenized_input['input_ids'])
    #         valid_data['type_ids'].append(tokenized_input['token_type_ids'])
    #         valid_data['labels'].append(line['label'])
    #
    # with open('Data/train.pkl', 'wb') as f:
    #     pickle.dump(train_data, f)
    # with open('Data/valid.pkl', 'wb') as f:
    #     pickle.dump(valid_data, f)

    with open("Data/train.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open("Data/valid.pkl", 'rb') as f:
        valid_data = pickle.load(f)

    return MyDataset(train_data), MyDataset(valid_data)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "hfl/chinese-roberta-wwm-ext-large")  # "nghuyong/ernie-1.0" "hfl/chinese-roberta-wwm-ext-large"
    model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-roberta-wwm-ext-large")  # "nghuyong/ernie-1.0"
    collator = MyCollator(tokenizer)

    add_tokens = ['[DESC]']
    num_added_toks = tokenizer.add_tokens(add_tokens)
    assert num_added_toks == len(add_tokens)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, collator


def train_iter(batch, model, optimizer, scheduler, device, update):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                labels=labels, return_dict=False)
    loss = outputs[0]
    loss.backward()
    dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM, async_op=False)
    loss_item = loss.item()
    if update:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return loss_item


def test(test_loader, model, device):
    y_pred = list()
    y_true = list()
    y_score = list()
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = F.softmax(outputs.logits, dim=1)
        _, cur_pred = torch.max(logits, 1)
        y_pred += cur_pred.cpu().tolist()
        y_true += labels.tolist()
        y_score += logits.cpu().tolist()

    y_true_onehot = list()
    for value in y_true:
        tmp = [0] * 5
        tmp[value] = 1
        y_true_onehot.append(tmp)
    return accuracy_score(y_true, y_pred), mrr_score(y_true_onehot, y_score)


def valid(valid_loader, model, device):
    y_pred = list()
    y_true = list()
    y_score = list()
    for batch in tqdm(valid_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = F.softmax(outputs.logits, dim=1)
        _, cur_pred = torch.max(logits, 1)
        y_pred += cur_pred.cpu().tolist()
        y_true += labels.tolist()
        y_score += logits.cpu().tolist()

    y_true_onehot = list()
    for value in y_true:
        tmp = [0] * 5
        tmp[value] = 1
        y_true_onehot.append(tmp)
    accuracy = torch.tensor(accuracy_score(y_true, y_pred), device=device)
    mrr = torch.tensor(mrr_score(y_true_onehot, y_score), device=device)
    dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
    dist.all_reduce(mrr, op=dist.ReduceOp.SUM)
    return accuracy.cpu().tolist(), mrr.cpu().tolist()


def train(gpu, args, train_dataset, valid_dataset, model, collator):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    if rank == 0:
        writer = SummaryWriter(comment=f'_{model_name}')
    device = torch.device(f'cuda:{device_no[gpu]}')
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_no[gpu]], find_unused_parameters=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collator,
                              sampler=train_sampler)  # , shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4 * train_batch_size, collate_fn=collator,
                              sampler=valid_sampler)

    total_num_update = int((len(train_loader) * (train_epoch + 1) / update_freq))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                             num_warmup_steps=total_num_update * 0.1,
                                                             num_training_steps=total_num_update)
    # scheduler = transformers.get_constant_schedule_with_warmup(optimizer=optimizer,
    #                                                            num_warmup_steps=warmup_updates)
    iteration = 0
    loss = 0
    for i in range(train_epoch):
        for batch in tqdm(train_loader):
            iteration += 1
            update = iteration % update_freq == 0
            loss += train_iter(batch, model, optimizer, scheduler, device, update=update)
            if rank == 0 and update:
                writer.add_scalar('train/loss', loss / update_freq, iteration / update_freq)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], iteration / update_freq)
                loss = 0

            if iteration % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    accuracy, mrr = valid(valid_loader, model, device)
                    if rank == 0:
                        writer.add_scalar('test/accuracy', accuracy / args.gpus, iteration)
                        writer.add_scalar('test/mrr', mrr / args.gpus, iteration)
                        if not os.path.exists(f"Model/{model_name}/checkpoint-{iteration}"):
                            os.makedirs(f"Model/{model_name}/checkpoint-{iteration}")
                        torch.save(model.module.state_dict(),
                                   f'Model/{model_name}/checkpoint-{iteration}/pytorch.bin')
                model.train()



if __name__ == '__main__':
    transformers.set_seed(42)

    # Train
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=nodes,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=gpus, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes，就是当前node的编号')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes  #
    os.environ['MASTER_ADDR'] = 'localhost'  #
    os.environ['MASTER_PORT'] = '123453'  #
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    print("Loading Model......")
    model, tokenizer, collator = load_model()
    print("Loading Data......")
    train_dataset, valid_dataset = load_data(tokenizer)
    mp.spawn(train, nprocs=args.gpus,
             args=(args, train_dataset, valid_dataset, model, collator))

    """single thread"""
    # writer = SummaryWriter(comment=f'_{model_name}')
    # device = torch.device(f'cuda:0')
    # model.to(device)
    # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collator)  # , shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=4 * train_batch_size, collate_fn=collator)
    # total_num_update = int((len(train_loader) * (train_epoch + 1) / update_freq))
    # optimizer = transformers.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
    #                                                          num_warmup_steps=total_num_update * 0.1,
    #                                                          num_training_steps=total_num_update)
    # iteration = 0
    # for i in range(train_epoch):
    #     for batch in tqdm(train_loader):
    #         iteration += 1
    #         loss = train_iter(batch, model, optimizer, scheduler, device)
    #         writer.add_scalar('train/loss', loss, iteration)
    #         writer.add_scalar('train/lr', scheduler.get_last_lr()[0], iteration)
    #
    #         if iteration % eval_every == 0:
    #             model.eval()
    #             with torch.no_grad():
    #                 accuracy, mrr = valid(valid_loader, model, device)
    #                 writer.add_scalar('test/accuracy', accuracy, iteration)
    #                 writer.add_scalar('test/mrr', mrr, iteration)
    #                 if not os.path.exists(f"Model/{model_name}/checkpoint-{iteration}"):
    #                     os.makedirs(f"Model/{model_name}/checkpoint-{iteration}")
    #                 torch.save(model.state_dict(),
    #                            f'Model/{model_name}/checkpoint-{iteration}/pytorch.bin')
    #             model.train()



    # # Test
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print("Loading Data......")
    # train_dataset, valid_dataset, test_dataset, _ = load_data()
    # print("Loading Model......")
    # model, tokenizer, collator = load_file(f'Model/{model_name}', 10)
    # test(test_dataset, model, tokenizer, collator, device)
