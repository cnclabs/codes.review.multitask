#!/bin/sh

gdown https://drive.google.com/u/1/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
gunzip GoogleNews-vectors-negative300.bin.gz 
mv GoogleNews-vectors-negative300.bin model.bin

wget http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.pkl.bz2
bunzip2 http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.pkl.bz2
