# KG4Ex

This is the official implementation for our paper **KG4Ex: An Explainable Knowledge Graph-Based Approach for Exercise Recommendation**, accepted by **CIKM'23**.


## Requirements
The code is built on Pytorch and the [pyKT](https://github.com/pykt-team/pykt-toolkit/tree/main) benchmark library. Run the following code to satisfy the requeiremnts by pip: `pip install -r requirements.txt`


## Datasets
- Download the three public datasets we use in the paper at:

  [ASSISTments 2009](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)

  [Algebra 2005](https://pslcdatashop.web.cmu.edu/KDDCup/)

  [Statics 2011](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507)

- Preprocess the dataset using [pyKT](https://github.com/pykt-team/pykt-toolkit/tree/main) to obtain the student's mastery level of knowledge concepts **(MLKC)**, the probability of knowledge concepts appearing in the next exercise **(PKC)**, and the forgetting rate of knowledge concepts **(FRKC)**.

- We provide an example of a CSV file obtained after [pyKT](https://github.com/pykt-team/pykt-toolkit/tree/main) processing using the [Algebra 2005](https://pslcdatashop.web.cmu.edu/KDDCup/) dataset (top 10 rows), located in folder `KG4Ex/pyKT_example/Algebra2005.csv`.


## Run KG4Ex

1. Construct the knowledge graph: use pyKT preprocessed files, for example, Algebra2005.csv, to construct `entities.dict` (entity dictionaries), `relations.dict` (relationship dictionaries), `triples.txt` (triples required for knowledge graphs) and `Q.txt` (Q-matrix). Place the three generated files in folder `KG4Ex/data/algebra2005`.

2. Embedding learning: `python run.py --do_train --cuda --data_path ../data/algebra2005 --model TransE -b 1024 -d 1000 -g 12.0 -a 1.0 -lr 0.001 -adv -save models/algebra2005/TransE_adv`.

3. Recommendation and evaluation: the embedding vectors of entities and relationships are saved in `KG4Ex/codes/models`. Run `test_TransE.py` to obtain corresponding indicator results.


## Citation
If you find our work helpful, please kindly cite our research paper:
```
@inproceedings{10.1145/3583780.3614943,
author = {Guan, Quanlong and Xiao, Fang and Cheng, Xinghe and Fang, Liangda and Chen, Ziliang and Chen, Guanliang and Luo, Weiqi},
title = {KG4Ex: An Explainable Knowledge Graph-Based Approach for Exercise Recommendation},
url = {https://doi.org/10.1145/3583780.3614943},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
pages = {597–607},
year = {2023},
}

```
