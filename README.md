# A Multi-task Learning Framework with Disposable Auxiliary Networks for Early Prediction of Product Success
## 1. Introduction

### 1.1. Requirements
- python3.X
- pytorch
- numpy
- gensim
- wikipedia2vec

### 1.2. Datasets
### 1.3. Pretrained Word Embeddings
There are two pre-trained word embeddins needed. One is for English and the other is for Japanese.

- English
https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
- Japanese
https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

### 1.4. Getting Started
#### Download
```
$ git clone
$ cd ./
```

#### Install python packages
```
$ pip install -r requirements.txt
```

## 2. Usage

### 2.1. Train a new model

### 2.2. Get the predicted score for a given movie

#### Parameters
```
$ python main.py -h
usage: Training [-h] [--gpu GPU] [--epoch EPOCH] [--batch BATCH]
                [--sample_size SAMPLE_SIZE] [--lang LANG] [--task TASK]
				[--train TRAIN]

Arguments

optional arguments:
	-h,							--help				show this help message and exit
	--gpu GPU,					-g GPU				-1=cpu, 0, 1,...= gpt
	--epoch EPOCH,				-epoch				EPOCH
	--batch BATCH,				-batch				BATCH batch size
	--sample_size SAMPLE_SIZE,	-sample				SAMPLE_SIZE
	--lang LANG,				-lang LANG			en=English, jp=Japanese
	--task TASK,				-task TASK			reg=Regression, rank=Ranking
	--train TRAIN,				-train TRAIN		path of training data
```
