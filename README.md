
# Adversarial Training Meothds for Network Embedding.

This is our implementation for the following paper:

>Quanyu Dai, Xiao Shen, Liang Zhang, Qiang Li, and Dan Wang (2019). Adversarial Training Methods for Network Embedding, [Paper in ACM DL](https://dl.acm.org/citation.cfm?id=3313445). In WWW'19 The World Wide Web Conference, San Francisco, CA, USA, May 13-17, 2019.

Author: Quanyu Dai (quanyu.dai at connect.polyu.hk)

## Introduction
We introduce a succinct and effective local regularization method, namely adversarial training, to network embedding so as to achieve model robustness and better generalization performance. Specifically, the adversarial training method is applied by defining adversarial perturbations in the embedding space with an adaptive L2 norm constraint that depends on the connectivity pattern of node pairs.

## Citation 
If you would like to use our code, please cite:
```
@inproceedings{AdvT4NE_WWW2019,
  author    = {Quanyu Dai and
               Xiao Shen and
               Liang Zhang and
               Qiang Li and
               Dan Wang},
  title     = {Adversarial Training Methods for Network Embedding},
  booktitle = {The World Wide Web Conference, {WWW} 2019, San Francisco, CA, USA,
               May 13-17, 2019},
  pages     = {329--339},
  year      = {2019}
}
```

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* python == 3.6.5
* tensorflow-gpu == 1.12.0 
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

## Examples to Run the code with DeepWalk as Base Model
The instruction of commands has been clearly stated in the code (see AdvT4NE/main.py).
* Citeseer
```
python main.py --input_net 'input/citeseer.mat' --dataset 'citeseer' --eps 1.1 --reg_adv 1.0 --adv 'grad' --embed_size 128 --lr 0.001 --batch_size 1024 --nepoch 50 --resultTxt 'result/deepwalk-citeseer.txt' --rep 'output/citeseer-rep-deepwalk.mat' --pretraining_nepoch 2 --task mcc --normalized 0 --negative 1 --adapt_l2 1 --base 'deepwalk'
```

* Cora
```
python main.py --input_net 'input/cora.mat' --dataset 'cora' --eps 0.9 --reg_adv 1.0 --adv 'grad' --embed_size 128 --lr 0.001 --batch_size 1024 --nepoch 50 --resultTxt 'result/deepwalk-cora.txt' --rep 'output/cora-rep-deepwalk.mat' --pretraining_nepoch 2 --task mcc --normalized 0 --negative 1 --adapt_l2 1  --base 'deepwalk'
```

* Wiki
```
python main.py --input_net 'input/wiki.mat' --dataset 'wiki' --eps 0.5 --reg_adv 1.0 --adv 'grad' --embed_size 128 --lr 0.001 --batch_size 1024 --nepoch 50 --resultTxt 'result/deepwalk-wiki.txt' --rep 'output/wiki-rep-deepwalk.mat' --pretraining_nepoch 2 --task mcc --normalized 0 --negative 1 --adapt_l2 1  --base 'deepwalk'
```

## About Evaluation
For node classification, we do not normalize the embedddings before evaluation.
For link prediction, we normalize the embeddings before evaluation.
