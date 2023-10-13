import pickle
from random import random
Process_Data_Path = "G:\PycharmProjects\Knowledge-Graph-4-VIS-Recommendation-main\data\Process_Data.json"


# with open("inference_results/TransE.pkl", "rb") as f:
#     dict_data = pickle.load(f)
#     print(dict_data)

# import torch
# print(torch.__version__)
#
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())

import json

r2id = open(r"G:\PycharmProjects\Knowledge-Graph-4-VIS-Recommendation-main\data\triplets\relations.dict",'w')
for i in range(101) :
    r2id.write(f"{i}\tAbtain{i}\n")
r2id.close()

e2id = open(r"G:\PycharmProjects\Knowledge-Graph-4-VIS-Recommendation-main\data\triplets\entities.dict",'w')
prediction_Concept = open(r"G:\PycharmProjects\Knowledge-Graph-4-VIS-Recommendation-main\data\triplets\prediction_Concept.txt",'w')

Us_dict = {}
Concept_dict = {}

with open(Process_Data_Path,'r',encoding="UTF-8") as load_file:
    json_obj = json.load(load_file)
    with open(f"./data/triplets/train.txt", "w") as f:
        for x in json_obj:
            for y in json_obj[x]:
                UsId = int(x)
                Ps = int(float(json_obj[x][y])*100)
                ConceptId = int(y)
                f.write(f"Student{UsId}\tAbtain{Ps}\tConcept{ConceptId}\n")
                if UsId not in Us_dict:
                    Us_dict[UsId] = len(Us_dict)+len(Concept_dict)
                    e2id.write(f"{Us_dict[UsId]}\tStudent{UsId}\n")
                if ConceptId not in Concept_dict:
                    Concept_dict[ConceptId] = len(Us_dict)+len(Concept_dict)
                    e2id.write(f"{Concept_dict[ConceptId]}\tConcept{ConceptId}\n")
                    prediction_Concept.write(str(ConceptId)+'\n')

e2id.close()
prediction_Concept.close()