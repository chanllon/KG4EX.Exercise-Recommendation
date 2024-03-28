import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

dict_path = f"../data/algebra2005"
embedding_path = f"./models/algebra2005/TransE_adv"

relation_embedding = np.load(f"{embedding_path}/relation_embedding.npy")
entity_embedding = np.load(f"{embedding_path}/entity_embedding.npy")


### 读取Q矩阵
Q = []
with open(f"{dict_path}/Q.txt", 'r') as file:
    i = 0
    for line in file:
        kc = line.strip().split(',')
        kc_int = [int(x) for x in kc]
        Q.append(kc_int)

with open(f"{dict_path}/entities.dict", 'r') as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(f"{dict_path}/relations.dict", 'r') as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)


dict_entity_embedding = {}
dict_relation_embedding = {}

for (k, v) in entity2id.items():
    dict_entity_embedding[k] = entity_embedding[v, :]

for (k, v) in relation2id.items():
    dict_relation_embedding[k] = relation_embedding[v, :]

def TransE(head, relation, tail, gamma=12.0):
    score = (head + relation) - tail
    score = gamma - np.linalg.norm(score, ord=2)
    return score

uid_mlkc_dict = {}
uid_pkc_dict = {}
uid_exfr_dict = {}
uid_rec_ex_dict = {}
with open(f"{dict_path}/test_triples.txt", 'r', encoding="UTF-8") as load_file:
    for line in load_file:
        item1, item2, uid = line.strip().split('\t')
        if item2[0] == 'm':
            kc, mlkc, uid = item1, item2, uid
            if uid not in uid_mlkc_dict.keys():
                uid_mlkc_dict[uid] = {}
            uid_mlkc_dict[uid][kc] = 'mlkc' + str(round(float(mlkc[4:]), 2))
        elif item2[0] == 'e':
            ex, exfr, uid = item1, item2, uid
            if uid not in uid_exfr_dict.keys():
                uid_exfr_dict[uid] = {}
            uid_exfr_dict[uid][ex] = 'exfr' + str(round(float(exfr[4:]), 2))
        else:
            kc, pkc, uid = item1, item2, uid
            if uid not in uid_pkc_dict.keys():
                uid_pkc_dict[uid] = {}
            uid_pkc_dict[uid][kc] = 'pkc' + str(round(float(pkc[3:]), 2))




### 计算推荐列表
uid_ex_scores = []
user_num = 0
rec_embedding = torch.from_numpy(dict_relation_embedding['rec'])
uid_mlkc_dict_keys_list = [key for key in uid_mlkc_dict.keys()]
print("start!!!")


# 遍历每个用户，得到每个用户对每道题的评分
for uid in uid_mlkc_dict_keys_list:
    user_num += 1
    uid_mlkc_keys = list(uid_mlkc_dict[uid].keys())
    uid_exfr_keys = list(uid_exfr_dict[uid].keys())
    uid_pkc_keys = list(uid_pkc_dict[uid].keys())
    print(f"************************start: {user_num} -- {uid}************************")
    scores = []

    s_mlkc_list = []
    s_pkc_list = []
    s_efr_list = []

    for i in range(len(uid_mlkc_keys)):
        kc_embedding = torch.from_numpy(dict_entity_embedding[uid_mlkc_keys[i]])
        mlkc_embedding = torch.from_numpy(dict_relation_embedding[uid_mlkc_dict[uid][uid_mlkc_keys[i]]])
        pkc_embedding = torch.from_numpy(dict_relation_embedding[uid_pkc_dict[uid][uid_pkc_keys[i]]])

        s_mlkc_list.append(kc_embedding + mlkc_embedding)
        s_pkc_list.append(kc_embedding + pkc_embedding)

    for qid in range(len(Q)):
        e = torch.from_numpy(dict_entity_embedding['ex' + str(qid)])
        fr1 = 0.0
        fr2 = 0.0

        for s_mlkc in s_mlkc_list:
            fr1 += TransE(s_mlkc, rec_embedding, e)

        for s_pkc in s_pkc_list:
            fr1 += TransE(s_pkc, rec_embedding, e)

        ej_embedding = torch.from_numpy(dict_entity_embedding[uid_exfr_keys[qid]])
        efr_embedding = torch.from_numpy(dict_relation_embedding[uid_exfr_dict[uid][uid_exfr_keys[qid]]])
        s_efr = ej_embedding + efr_embedding
        fr2 = TransE(s_efr, rec_embedding, e)

        O_sel = fr1 / len(uid_mlkc_keys) + fr2
        scores.append(O_sel)

    uid_ex_scores.append((uid, scores))
    print(f"************************finish: {user_num} -- {uid}************************")


print("-----------------------------------------------开始计算acc-----------------------------------------------")

def ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n):
    acc = []
    for item in uid_ex_scores:
        # 将得分列表与qid一起存储，并进行排序
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]  # 取得分前n位的习题进行推荐
        user_mlkc = uid_mlkc_dict[uid]
        diff = 0
        for ex_id in uid_ex_score:
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            ex_ml = 1.0
            for kc in kc_list:
                ex_ml = ex_ml * float(user_mlkc['kc' + str(kc)][4:])
            diff += 1 - np.abs(r1 - (ex_ml))
        acc.append(diff / n)
    return np.mean(acc), np.std(acc)

r1 = 0.7
for n in [10]:
    mean_acc, std_acc = ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n)
    print(f"The recommendation list length is n = {n}, the mean ACC = {mean_acc}, the std ACC = {std_acc}")