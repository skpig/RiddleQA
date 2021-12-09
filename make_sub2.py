import json
import requests
from tqdm import tqdm

# http://shuyantech.com/api/cndbpedia/ment2ent?q= #! 根据单词返回实体列表
# http://shuyantech.com/api/cndbpedia/avpair?q= #! 输入实体名返回三元知识(word)
# http://shuyantech.com/api/cnprobase/ment2ent?q= #! 输入单词返回实体列表
# http://shuyantech.com/api/cnprobase/concept?q= #! 输入实体返回实体概念列表
# http://shuyantech.com/api/entitylinking/cutsegment?q= #!分词筛选出现的概念列表


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
                d_line = json.loads(line) #json按照dict形式存储
                # 对于choice,需要对于每一个list获得entity
                choice_list = d_line["choice"] # choice列表
                label = d_line["label"] # label
                entity_list = [] # choice对应的entity列表
                desc_list = [] # choice对应的entity的desc列表

                entity2_list = []
                desc2_list = []

                cnt = 0
                for c in tqdm(choice_list):
                    c_entityL = word2entityL(c)
                    
                    c_descL = []
                    c2_descL = []

                    # 获得entity列表
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

                    # 获得desc列表
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

# 统计结果
with open("INFO.txt", "w", encoding='utf-8') as f:
    str = "noEntity_cnt =" + str(noEntity_cnt) + "\n" + "noDesc_cnt =" + str(noDesc_cnt) + "\n" + \
        "noEntityAns_cnt =" + str(noEntityAns_cnt) + "\n" + "noDescAns_cnt =" + str(noDescAns_cnt) + "\n" + \
        "total =" + str(n)

    f.write(str)
