import json

import numpy as np

dict_path = f"../data/triplets"
embedding_path = f"../embeddings/TransE1"
Process_Data_Path = "G:\PycharmProjects\Knowledge-Graph-4-VIS-Recommendation-main\data\Process_Data.json"
entity_embedding = np.load(f"{embedding_path}/entity_embedding.npy")
relation_embedding = np.load(f"{embedding_path}/relation_embedding.npy")

prediction_excise = []
with open(r"G:\PycharmProjects\Knowledge-Graph-4-VIS-Recommendation-main\data\triplets\prediction_Concept.txt",'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        prediction_excise.append("Concept"+line)



with open(f"{dict_path}/entities.dict") as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(f"{dict_path}/relations.dict") as fin:
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


del entity_embedding, relation_embedding

def TransE(head, relation, tail, gamma=24):

    score = (head + relation) - tail

    score = gamma - np.linalg.norm(score, ord=2)
    return score

hit_num = np.array([0 for i in range(10)])
averange_rank = 0
hit_total_num = 0

with open(Process_Data_Path,'r',encoding="UTF-8") as load_file:
    json_obj = json.load(load_file)
    for UsId in json_obj:
        h = dict_entity_embedding["Student" + UsId]
        for PS in json_obj[UsId]:
            hit_total_num += 1;
            r = dict_relation_embedding["Abtain" + str(int(float(json_obj[UsId][PS])*100))]
            t = dict_entity_embedding["Concept" + PS]
            try:
                Array_Concept_Score = np.array([TransE(h, r, dict_entity_embedding[i], \
                                     gamma=12.0) for i in prediction_excise])
                Array_Concept_Score = sorted(Array_Concept_Score,reverse=True)
                tempt_rank = Array_Concept_Score.index(TransE(h, r, t, gamma=12.0))
                if tempt_rank <= 9:
                    hit_num[tempt_rank:] += 1
                averange_rank += tempt_rank
            except:
                continue

print(f"hit2_num: {hit_num}\nhit_total_num: {hit_total_num}\nHits@: {hit_num/hit_total_num}")
print(averange_rank/hit_total_num)

