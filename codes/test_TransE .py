import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

dict_path = f"../data/algebra2005"
embedding_path = f"./models/algebra2005/TransE_adv"

relation_embedding = np.load(f"{embedding_path}/relation_embedding.npy")
entity_embedding = np.load(f"{embedding_path}/entity_embedding.npy")


### read Q-matrix, entities, relations
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


### read embeddings
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


### read test_triples
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



### calculate the score of each user for each exercise
uid_ex_scores = []
user_num = 0
rec_embedding = torch.from_numpy(dict_relation_embedding['rec'])
uid_mlkc_dict_keys_list = [key for key in uid_mlkc_dict.keys()]
print("start!!!")


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

def ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n):
    acc = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
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

def Nov(uid_kc_response, uid_ex_scores, Q, n):
    jaccsim = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
        kc_response = set(uid_kc_response[uid])
        jaccard_similarity = 0
        for ex_id in uid_ex_score:
            rec_ex_kc_set = set()
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            rec_ex_kc_set.update(kc_list)
            intersection = len(kc_response.intersection(rec_ex_kc_set))
            union = len(kc_response.union(rec_ex_kc_set))
            jaccard_similarity += 1 - intersection / union
        jaccsim.append(jaccard_similarity / n)
    return np.mean(jaccsim), np.std(jaccsim)


print("-----------------------------------------------Start calculating ACC-----------------------------------------------")
r1 = 0.7
for n in [10]:
    mean_acc, std_acc = ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n)
    print(f"The recommendation list length is n = {n}, the mean ACC = {mean_acc}, the std ACC = {std_acc}")


print("-----------------------------------------------Start calculating NOV-----------------------------------------------")
all_uid_kc_response = {}
with open(f"{dict_path}/algebra2005_uid_kc_response.txt", 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        uid = line[0]
        correct_kc_response = [int(x) for x in line[1].split(',')]
        all_uid_kc_response[uid] = correct_kc_response

test_uid_kc_response = {}
for uid in uid_mlkc_dict.keys():
    test_uid_kc_response[uid] = all_uid_kc_response[uid]

for n in [10]:
    mean_nov, std_nov = Nov(test_uid_kc_response, uid_ex_scores, Q, n)
    print(f"The recommendation list length is n = {n}, the mean NOV = {mean_nov}, the std NOV = {std_nov}")
