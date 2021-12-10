import json
import re

import jsonlines
import requests
from tqdm import tqdm
import jieba
import jieba.analyse


# http://shuyantech.com/api/cndbpedia/ment2ent?q= #! 根据单词返回实体列表
# http://shuyantech.com/api/cndbpedia/avpair?q= #! 输入实体名返回三元知识(word)
# http://shuyantech.com/api/cnprobase/ment2ent?q= #! 输入单词返回实体列表
# http://shuyantech.com/api/cnprobase/concept?q= #! 输入实体返回实体概念列表
# http://shuyantech.com/api/entitylinking/cutsegment?q= #!分词筛选出现的概念列表

stopwords = list()
with open("Data/hit_stopwords.txt", 'r') as f:
    for line in f.readlines():
        stopwords.append(line.strip())
stopwords = set(stopwords)




# 输入单词或短语,返回实体列表列表(取代http://shuyantech.com/api/entitylinking/cutsegment?q=的更精确功能)
def word2entityL(w):
    url = "http://shuyantech.com/api/cndbpedia/ment2ent?q=" + w + "&apikey=6474c03336e13f7d130ab3eadff84b8"
    r = requests.get(url)
    cnt = 0
    while (r.status_code != 200 and cnt < 5):
        r = requests.get(url)
        cnt += 1
    e_list = r.json()["ret"]
    return e_list


def entity2desc(e):
    # ensure the entity can be find in KG
    url = "http://shuyantech.com/api/cndbpedia/avpair?q=" + e + "&apikey=6474c03336e13f7d130ab3eadff84b8"
    r = requests.get(url)
    cnt = 0
    while (r.status_code != 200 and cnt < 5):
        r = requests.get(url)
        cnt += 1
    rw_list = r.json()["ret"]
    for rw in rw_list:
        if rw[0] == "DESC":
            return rw[1]
    # 没有的话.需要统计比例,并用另一种内容替代
    return ""





def get_entity_and_desc(mode):
    noEntity_cnt = 0
    noDesc_cnt = 0
    noEntityAns_cnt = 0
    noDescAns_cnt = 0
    with jsonlines.open(f"Data/{mode}.jsonl", 'r') as reader:
        with jsonlines.open(f"Data/{mode}_raw.jsonl", "w") as writer:
            for d_line in tqdm(reader):
                # 对于choice,需要对于每一个list获得entity
                choice_list = d_line["choice"]  # choice列表
                label = d_line["label"]  # label
                entity_list = []  # choice对应的entity列表
                desc_list = []  # choice对应的entity的desc列表

                cnt = 0
                for choice in tqdm(choice_list):
                    c_entityL = word2entityL(choice)
                    c_descL = []

                    # 获得entity列表
                    if len(c_entityL) == 0:
                        # if find no entity, cut word into small pieces, and search again
                        word_list = jieba.lcut_for_search(choice)
                        word_list += list(choice)
                        for word in word_list:
                            if word == choice:
                                continue
                            c_entityL += word2entityL(word)
                        # if still no entity, record it
                        if len(c_entityL) == 0:
                            noEntity_cnt += 1
                            print("NoEntity: ", choice)
                            if cnt == label:
                                noEntityAns_cnt += 1

                    entity_list.append(c_entityL)

                    # 获得desc列表
                    flag = True
                    for e in c_entityL:
                        e_descL = entity2desc(e)  # entity desc
                        if len(e_descL) == 0:  # 条目不存在desc时, 直接跳过
                            continue
                        c_descL.append(e_descL)
                        flag = False

                    if flag:  # 如果所有条目都不存在desc
                        noDesc_cnt += 1
                        if cnt == label:
                            noDescAns_cnt += 1
                            print("AnsNoDeac: ", choice)

                    desc_list.append(c_descL)
                    cnt += 1

                d_line["entities"] = entity_list
                d_line["desc"] = desc_list

                writer.write(d_line)
    log_info = "noEntity_cnt =" + str(noEntity_cnt) + "\n" + "noDesc_cnt =" + str(noDesc_cnt) + "\n" + \
             "noEntityAns_cnt =" + str(noEntityAns_cnt) + "\n" + "noDescAns_cnt =" + str(noDescAns_cnt) + "\n"
    print(log_info)
    # # 统计结果
    # with open("INFO.txt", "w", encoding='utf-8') as f:
    #     string = "noEntity_cnt =" + str(noEntity_cnt) + "\n" + "noDesc_cnt =" + str(noDesc_cnt) + "\n" + \
    #           "noEntityAns_cnt =" + str(noEntityAns_cnt) + "\n" + "noDescAns_cnt =" + str(noDescAns_cnt) + "\n" + \
    #           "total =" + "4000"
    #
    #     f.write(string)

# 验证desc是否相关，可调整（词袋/字袋，重叠个数，比例阈值）
def verify_desc(decs, query):
    query_BoW = set(jieba.lcut_for_search(query))  # 维护词袋
    # 对每一个desc进行分句
    sentences = re.split(r'[，。；\n\s]\s*', decs)
    filtered_sentences = list()
    for sentence in sentences:
        is_similar = False
        for word in query_BoW:
            if sentence.find(word) != -1:
                is_similar = True
                break
        if is_similar:
            filtered_sentences.append(sentence)
    return len(filtered_sentences) / len(sentences) > 0.1, '，'.join(filtered_sentences)

def verify_entity(entity, hint):
    # 即entity符合hint
    if entity.find(hint) != -1:
        return True

    url = "http://shuyantech.com/api/cnprobase/concept?q=" + entity + "&apikey=6474c03336e13f7d130ab3eadff84b8"
    r = requests.get(url)
    cnt = 0
    while (r.status_code != 200 and cnt < 5):
        r = requests.get(url)
        cnt += 1
    rw_list = r.json()["ret"]
    for rw in rw_list:
        # 即entity的上位概念含有hint
        if rw[0].find(hint) != -1:
            return True
    return False

def filter_desc(mode):
    # from sentence_transformers import SentenceTransformer, util
    # model = SentenceTransformer('distiluse-base-multilingual-cased')

    no_entity_choice_num = 0
    no_entity_answer_num = 0
    with jsonlines.open(f"Data/{mode}_raw.jsonl", 'r') as r, \
            jsonlines.open(f"Data/{mode}.jsonl", 'w') as w:
        for line in tqdm(r):
            for choice_idx, descs in enumerate(line['desc']):

                # 说明是全词匹配，无需计算相似度，直接保留
                if len(descs) == 1:
                    continue

                valid_enity_ids = []  # 相关entity的标号
                for entity_idx, desc in enumerate(descs):
                    entity = line['entities'][choice_idx][entity_idx]
                    # 如何entity符合hint的要求，则直接保留
                    if verify_entity(entity, line['hint']):
                        valid_enity_ids.append(entity_idx)
                        break

                    is_similar, filtered_desc = verify_desc(entity + desc, line['riddle'])
                    # 如果该entity相关
                    if is_similar:
                        valid_enity_ids.append(entity_idx)
                        descs[entity_idx] = filtered_desc
                if len(valid_enity_ids) == 0:
                    line['entities'][choice_idx] = ['无']
                    line['desc'][choice_idx] = ['无']
                    no_entity_choice_num += 1
                    if choice_idx == line['label']:
                        no_entity_answer_num += 1
                else:
                    line['entities'][choice_idx] = [line['entities'][choice_idx][ids] for ids in valid_enity_ids]
                    line['desc'][choice_idx] = [line['desc'][choice_idx][ids] for ids in valid_enity_ids]
            w.write(line)
        print(f"Total {no_entity_choice_num} choice doesn't have entity")
        print(f"Total {no_entity_answer_num} label choice doesn't have entity")


if __name__ == "__main__":
    # get_entity_and_desc('valid')
    filter_desc('valid')
