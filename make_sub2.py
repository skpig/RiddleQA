import json
from os import terminal_size
import pickle
import re

import jieba
import requests
from tqdm import tqdm

# http://shuyantech.com/api/cndbpedia/ment2ent?q= #! 根据单词返回实体列表
# http://shuyantech.com/api/cndbpedia/avpair?q= #! 输入实体名返回三元知识(word)
# http://shuyantech.com/api/cnprobase/ment2ent?q= #! 输入单词返回实体列表
# http://shuyantech.com/api/cnprobase/concept?q= #! 输入实体返回实体概念列表
# http://shuyantech.com/api/entitylinking/cutsegment?q= #!分词筛选出现的概念列表

# 方案2:对于hint,手动筛选归类一下
# 对于答案:DESC词条用于匹配,其余词条对应HINT(本身可也以用于对应)

# reg = "[\[\],.;\']"
# def remove_Punctuation(text):
#     text = re.sub(reg,' ',text)
#     return text.strip()



# def get_StopWord():
#     with open("stopword.pkl", "rb") as f:
#         l = pickle.load(f)
#         f.close()
#     return l

# stopWord = get_StopWord()

# def remove_StopWord(l):
#     l1 = []
#     for w in l:
#         if w in stopWord:
#             pass
#         else:
#             l1.append(w)
#     return l1

# 判断1:只考虑同一层实体,实体与实体进行匹配

# 输入单词或短语,返回实体列表列表(取代http://shuyantech.com/api/entitylinking/cutsegment?q=的更精确功能)
def word2entityL(w):
    url = "http://shuyantech.com/api/cndbpedia/ment2ent?q=" + w + "&apikey=6474c03336e13f7d130ab3eadff84b8"
    r = requests.get(url)
    cnt = 0
    while (r.status_code != 200 and cnt < 5):
        r = requests.get(url)
        cnt += 1
    e_list =  r.json()["ret"]
    if len(e_list):
        return e_list
    return []

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
        


def is_Chinese(w):
    for ch in w:
        if '\u4e00' > ch or ch > '\u9fff':
            return False
    return True

# 输入单词或短语列表,返回实体列表
def wordL2entityL(w_l):
    e_list = []
    for w in w_l:
        if is_Chinese(w):
            e_list = e_list + word2entityL(w)
    return list(set(e_list))

# 输入一个实体,返回下一层单词列表
def entity2NwordL(e, mode=0):
    url = "http://shuyantech.com/api/cndbpedia/avpair?q=" + e + "&apikey=6474c03336e13f7d130ab3eadff84b8"
    rw_l = requests.get(url)
    rw_l = rw_l.json()

    w_l = []
    for w in rw_l["ret"]:
        if mode == 0:
            w_l.append(w[1])
        elif mode == 1:
            if(len(w[1]) < 15):
                w_l.append(w[1])
    return list(set(w_l))

# 输入一个实体,返回下一层实体列表
def entity2NentityL(e, mode=0):
    w_l = entity2NwordL(e, mode)
    e_l = wordL2entityL(w_l)
    return list(set(e_l))

# 输入实体列表,返回下一层实体列表
def entityL2NentityL(e_l):
    e_list = []
    e_listR = []
    for e in tqdm(e_l):
        e_listR = e_listR + entity2NentityL(e, 0)
        e_list = e_list + entity2NentityL(e, 1)
    return list(set(e_list)), list(set(e_listR))

# 对于超长词汇的处理:不进行下一步的搜索,但是分词,保留在当前列表中

ban = ["汉语", "词语", "词汇", "字词", "汉字", "语言"]

n = 4000
with open("test.jsonl", encoding='utf-8') as f1:
    with open("new2.json", "w", encoding='utf-8') as f3: 
        with open("new.json", "w", encoding='utf-8') as f2:
            noEntity_cnt = 0
            noDesc_cnt = 0
            noEntityAns_cnt = 0
            noDescAns_cnt = 0
            for i in tqdm(range(n)):
                line = f1.readline()
                d_line = json.loads(line)
                # 对于choice,需要对于每一个list获得entity
                choice_list = d_line["choice"]
                label = d_line["label"]
                entity_list = []
                desc_list = []

                entity2_list = []
                desc2_list = []

                cnt = 0
                for c in tqdm(choice_list):
                    c_entityL = word2entityL(c)
                    
                    c_descL = []
                    c2_descL = []

                    if len(c_entityL) == 0:
                        noEntity_cnt += 1
                        if cnt == label:
                            noEntityAns_cnt += 1
                            print("AnsNoEntity: ", c)
                        entity2_list.append(["pad"])
                        entity_list.append(c_entityL)
                    else:
                        entity2_list.append(c_entityL)
                        entity_list.append(c_entityL)

                    flag = True
                    for e in c_entityL:
                        e_descL= entity2desc(e) # entity desc
                        if len(e_descL) == 0: # 条目不存在desc时
                            c_descL.append(e_descL) # 加入空字符串
                            c2_descL.append("pad") # 用pad替换
                        else:
                            c_descL.append(e_descL)
                            c2_descL.append(e_descL)
                            flag = False
                        
                    if flag: # 如果所有条目都不存在desc
                        noDesc_cnt += 1
                        if cnt == label:
                            noDescAns_cnt += 1
                            print("AnsNoDeac: ", c)

                    desc_list.append(c_descL)
                    desc2_list.append(c2_descL)
                    cnt += 1
                
                d_line["entities"] = entity_list
                d_line["desc"] = desc_list
                json.dump(d_line, f2, ensure_ascii=False)
                f2.write('\n')

                d_line["entities"] = entity2_list
                d_line["desc"] = desc2_list
                json.dump(d_line, f3, ensure_ascii=False)   
                f3.write('\n')             


with open("INFO.txt", "w", encoding='utf-8') as f:
    str = "noEntity_cnt =" + str(noEntity_cnt) + "\n" + "noDesc_cnt =" + str(noDesc_cnt) + "\n" + \
        "noEntityAns_cnt =" + str(noEntityAns_cnt) + "\n" + "noDescAns_cnt =" + str(noDescAns_cnt) + "\n" + \
        "total =" + str(n)

    f.write(str)