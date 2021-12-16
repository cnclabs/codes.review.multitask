# A Multi-task Learning Framework with Disposable Auxiliary Networks for Early Prediction of Product Success
## 1. Introduction
Consider the scenario in which an investor seeks to iden- tify potential products before they are unveiled to the public. For such a scenario, the investor may pose questions such as “What characteristic better represents a product?” or “What features make a product popular?” Unlike traditional recom- mendation problems, in this case, there is no user feedback for such upcoming products, which makes associated predic- tion extremely challenging. To address this challenging yet common scenario, in this paper, we present a multi-task learn- ing framework that trains the prediction model on information for mature products that have user feedback, and then uses the model to predict the success of upcoming products with- out any user feedback. To achieve this goal, the framework consists of a main task network to extract product features from their descriptions and a novel disposable auxiliary net- work that learns domain-specific words and popular trends from user reviews at the same time. This disposable auxiliary network is beneficial during the training of the main task net- work, and is unused at the inference stage. Empirical results on two real-world datasets demonstrate that this multi-task learning framework not only significantly improves the over- all rating prediction for products but also effectively identifies the top successful products without any user reviews.


### 1.1. Requirements
- python3.X
- pytorch
- numpy
- gensim
- wikipedia2vec

### 1.2. Datasets
We provide two dataset, IMDB and filmark.
```
|--data/
  |--imdb
  |--filmark
  |--sample
```

The dataset of IMDB is used for English and the other one is crawled from filmark for Japanese.

### 1.3. Pretrained Word Embeddings
There are two pre-trained word embeddins needed. One is for English and the other is for Japanese.

- English
https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
- Japanese
https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
#### Download 
```
$ bash get_pkl.sh
```

### 1.4. Getting Started
#### Download
```
$ git clone
$ cd ./
```

## 2. Usage

### 2.1. Train a new model

#### Parameters
```
$ cd ./code
$ python main.py -h
usage: Training [-h] [--gpu GPU] [--epoch EPOCH] [--batch BATCH]
                [--sample_size SAMPLE_SIZE] [--lang LANG] [--task TASK]
				[--train TRAIN]

Arguments

optional arguments:
	-h,		--help	show this help message and exit
	--gpu GPU,	-g GPU	-1=cpu, 0, 1,...= gpt
	--epoch EPOCH,	-epoch	EPOCH
	--batch BATCH,	-batch	BATCH batch size
	--sample_size SAMPLE_SIZE,	-sample	SAMPLE_SIZE
	--lang LANG,	-lang LANG	en=English, jp=Japanese
	--task TASK,	-task TASK	reg=Regression, rank=Ranking
	--train TRAIN,	-train TRAIN	path of training data
```
