device_no = [0,1,2,3]
gpus = 4
assert gpus == len(device_no)
nodes = 1
model_name = 'roberta'  # 'ernie' 'roberta'
"""Train"""
eval_every = 100
save_every = 500
train_epoch = 20

dropout = 0.1

train_batch_size = 4
batch_size = 32
update_freq = int(batch_size / (train_batch_size * gpus))
assert update_freq > 0

# optimizer
lr = 5e-06
weight_decay = 0.1

#scheduler
warmup_updates = 600
total_num_update = None
clip_norm = 0.1


"""Test"""
test_batch_size = 10
