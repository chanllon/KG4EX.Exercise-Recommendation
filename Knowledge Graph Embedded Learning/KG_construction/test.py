import pickle
with open('./categorical_relation_mapping.pkl','rb') as path:
    print(pickle.load(path))

with open('../feature_extraction/feature_list_float_bool.pkl','rb') as path:
    print(pickle.load(path))