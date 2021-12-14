import json
from numpy.lib.twodim_base import fliplr
import requests
from tqdm import tqdm
import jieba
import jsonlines

# http://shuyantech.com/api/cndbpedia/ment2ent?q= #! 根据单词返回实体列表
# http://shuyantech.com/api/cndbpedia/avpair?q= #! 输入实体名返回三元知识(word)
# http://shuyantech.com/api/cnprobase/ment2ent?q= #! 输入单词返回实体列表
# http://shuyantech.com/api/cnprobase/concept?q= #! 输入实体返回实体概念列表
# http://shuyantech.com/api/entitylinking/cutsegment?q= #!分词筛选出现的概念列表


# 输入单词或短语,返回实体列表列表(取代http://shuyantech.com/api/entitylinking/cutsegment?q=的更精确功能)
def word2entityL(w):
    url = "http://shuyantech.com/api/cndbpedia/ment2ent?q=" + \
        w + "&apikey=6474c03336e13f7d130ab3eadff84b8"
    r = requests.get(url)
    cnt = 0
    while (r.status_code != 200 and cnt < 5):
        r = requests.get(url)
        cnt += 1
    e_list = r.json()["ret"]
    if len(e_list):
        return e_list
    return []


def entity2desc(e):
    # ensure the entity can be find in KG
    url = "http://shuyantech.com/api/cndbpedia/avpair?q=" + \
        e + "&apikey=6474c03336e13f7d130ab3eadff84b8"
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
    return "无"
# filter 字词,语言,名词

stopConcept = ["字词", "词语"]

def get_hint(e):
    url = "http://shuyantech.com/api/cnprobase/concept?q=" + \
        e + "&apikey=6474c03336e13f7d130ab3eadff84b8"
    r = requests.get(url)
    cnt = 0
    while (r.status_code != 200 and cnt < 5):
        r = requests.get(url)
        cnt += 1
    e_list = r.json()["ret"]
    e_return = []
    if len(e_list):
        for x in e_list:
            if x[0] not in stopConcept:
                e_return.append(x[0])
        return e_return
    return [] 


def get_hint_complex(ce) :
    hl = get_hint(ce)
    if len(hl) > 0:
        return hl
    else:
        return ["无"]


n = 500
with open("valid.jsonl", "r", encoding= 'utf-8') as f1,\
        open("valid_hint_why.json", "w",encoding= 'utf-8') as f2:

    for i in tqdm(range(n)):

        line = f1.readline()
        d_line = json.loads(line)  # json按照dict形式存储
        # 对于choice,需要对于每一个list获得entity

        choices = d_line["choice"]  # choice列表
        label = d_line["label"]  # lab
        hint = d_line["hint"]
        entity_final = []  # choice对应的entity列表
        desc_final = []
        hc_final = []
        in_hint = []

        # get hint concept,会有一个或者多个,但是如果hint直接能够搜索到concept,视为hint已经被选择出
        flag_hint = -1 # 选定hint的角标
        hint_concepts = [] # 最终根据hint_flag 选择元素
        hint_concept = get_hint(hint)

        # 已经可以确定
        if len(hint_concept) > 0:
            hint_concepts.append(hint_concept)
            flag_hint = 0

        # 不可以确定,需要分词后检索
        else:
            f = 0
            tmp = []
            hw_l = jieba.lcut_for_search(hint)
            he_l = []
            for hw in hw_l:
                # 如果所有都加入的话,那么在check的时候需要选择出最合适的hint,方法是什么呢?[如果得分根据某两个最高,那么直接认为是这样的][如果没有利用到这一条规则的choice,则选择label]
                # 一维列表
                he_l += word2entityL(hw)

            for he in he_l:
                if len(get_hint(he)) > 0:
                    tmp.append(get_hint(he))
                    f = 1
            
            if f == 0:
                hint_concepts.append(["无"])
            else:
                hint_concepts = tmp

        entityconcept_final = []
        for c_idx, c in tqdm(enumerate(choices)):
            # get entities for each choice
            flag_entity = 0 # 选择哪一个实体作为结果
            flag_in = -1 # 判断是否属于hint
            
            c_entities = []
            tmp = word2entityL(c)
            if len(tmp) > 0:
                c_entities = tmp
            else:
                # cut word for null entites try to get extra entities
                cw_l = jieba.lcut_for_search(c)
                for cw in cw_l:
                    c_entities += word2entityL(cw)
                if len(c_entities) == 0:
                    c_entities += ["无"] # 专门处理,可以预计到后面的结果了,设置为"无"
            
            # 获得实体的DESC
            c_descs = [entity2desc(ce) for ce in c_entities] 
            # 获得实体的concept,即使为空,也进行保留,因为实体是可以查询的最小单元,不应再次分词
            c_concepts = [get_hint_complex(ce) for ce in c_entities] # 是二维数组



            # 选择,分为flag_hint == -1 与 flag_hint >= 0 两种情况
            if flag_hint == -1:
                hw_l = jieba.lcut_for_search(hint)
                hw_l.append(hint)
                for hw in hw_l:
                    if hw in c:
                        flag_in = 1

                for cc_idx, c_concept in enumerate(c_concepts):
                    # hint lv1 hint及其分词在出现在entity上位概念中
                    # entity lv1 entity的上位概念出现在hint及其分词中


                    if len(set(hw_l).intersection(set(c_concept))) > 0:
                        flag_in = 1
                        flag_entity = cc_idx
                        break
                    
                    # hint lv2 hint及其分词在出现在desc上位概念中
                    # entity lv2 entity的desc中出现了hint及其分词
                    desc = c_descs[cc_idx]

                    fl = 0
                    for hw in hw_l:
                        if hw in desc:
                            flag_in = 1
                            flag_entity = cc_idx
                            fl = 1
                            break
                    
                    if fl:
                        break

                    
                    # hint lv3 hint某一上位概念和desc中出现
                    # entity lv3 desc中出现hint描述的上位概念
                    for hc_idx, h_concept in enumerate(hint_concepts):
                        fl = 0
                        for hc in h_concept:
                            if hc in desc:
                                flag_in = 1
                                flag_entity = cc_idx
                                flag_hint = hc_idx
                                fl = 1
                                break
                        if fl:
                            break

                        # hint/entity lv4 上位概念重合
                        if len(set(h_concept).intersection(set(c_concept))) > 0:
                            flag_in = 1
                            flag_entity = cc_idx
                            flag_hint = hc_idx
                            break
                    if fl:
                        break

                if flag_in == -1:
                    flag_in = 0

        
            elif flag_hint >= 0:
                hw_l = jieba.lcut_for_search(hint)
                hw_l.append(hint)
                for hw in hw_l:
                    if hw in c:
                        flag_in = 1

                
                for cc_idx, c_concept in enumerate(c_concepts):
                    # hint lv1 hint及其分词在出现在entity上位概念中
                    # entity lv1 entity的上位概念出现在hint及其分词中
                    if len(set(hw_l).intersection(set(c_concept))) > 0:
                        flag_in = 1
                        flag_entity = cc_idx
                        break
                    
                    # hint lv2 hint及其分词在出现在desc上位概念中
                    # entity lv2 entity的desc中出现了hint及其分词
                    desc = c_descs[cc_idx]

                    fl = 0
                    for hw in hw_l:
                        if hw in desc:
                            flag_in = 1
                            flag_entity = cc_idx
                            fl = 1
                            break
                    if fl:
                        break
                    
                    # hint lv3 hint某一上位概念和desc中出现
                    # entity lv3 desc中出现hint描述的上位概念
                    # 因为已经确定了hint,那么可以直接决定了

                    fl = 0
                    for hc in hint_concepts[flag_hint]:
                        if hc in desc:
                            flag_in = 1
                            flag_entity = cc_idx
                            fl = 1
                            break
                    
                    if fl:
                        break

                        # hint/entity lv4 上位概念重合
                    if len(set(hint_concepts[flag_hint]).intersection(set(c_concept))) > 0:
                        flag_in = 1
                        flag_entity = cc_idx
                        break

                if flag_in == -1:
                    flag_in = 0
                    flag_entity = 0

            entity_final.append(c_entities[flag_entity])
            entityconcept_final.append(c_concepts[flag_entity])
            desc_final.append(c_descs[flag_entity])
            in_hint.append(flag_in)
            
        if flag_hint == -1:
            flag_hint = 0 # 默认hint下标为0
        d_line["entities"] = entity_final
        d_line["hint_concept"] = hint_concepts[flag_hint]
        d_line["entity_concept"] = entityconcept_final

        d_line["in_hint"] = in_hint
        d_line["desc"] = desc_final
        
        json.dump(d_line, f2, ensure_ascii=False)   
        f2.write('\n')


