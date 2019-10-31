import args
import evaluation as eva
import json
import random
import nltk
import numpy as np
import os
import sys
import torch
import torch_model as models
import training_handler
import util
from collections import Counter

def RUN(testing):
	baseline_rmse = []
	baseline_rank_rmse = []
	cnn_rmse = []
	cnn_rank_rmse = []
	rnn_rmse = []
	rnn_rank_rmse = []
	satt_rmse = []
	satt_rank_rmse = []
	attention_rmse = []

	test_loader = thandler.torch_testing_data(thandler.load_data(testing, 'test'))
	'''
	print(len(train_loader))
	tmp_ = []
	for i in train_loader:
		tmp_.extend(i[1].tolist())
	print(Counter(tmp_))
	tmp_ = []
	for i in test_loader:
		tmp_.extend(i[1].tolist())
	print(Counter(tmp_))
	'''	

	_, MLP_pred, _ = thandler.test( models.baseline(), test_loader, model_name='MLP.pkl' )
	_, MLP_Rank_pred, _ = thandler.test( models.RN_org(), test_loader, model_name='RN.pkl' )
	print('MLP')
	#print(MLP_pred, MLP_Rank_pred)
	#MLP_pred = list(map(lambda m: 1.0 if m > 0.5 else 0.0, MLP_pred ))
	#MLP_Rank_pred = list(map(lambda m: 1.0 if m > 0.5 else 0.0, MLP_Rank_pred ))
	#MLP_pred = list(map(torch.sigmoid, MLP_pred))
	#ML__Rank_pred = list(map(torch.sigmoid, MLP_Rank_pred))
	print(MLP_pred, MLP_Rank_pred)

	print('CNN')
	_, CNN_pred, _ = thandler.test( models.CNN(), test_loader, model_name='CNN.pkl' )
	_, CNN_Rank_pred, _ = thandler.test( models.CNN_Rank(), test_loader, model_name='CNN_Rank.pkl' )
	#print(CNN_pred, CNN_Rank_pred)
	#CNN_pred = list(map(lambda m: 1.0 if m > 0.5 else 0.0, CNN_pred ))
	#CNN_Rank_pred = list(map(lambda m: 1.0 if m > 0.5 else 0.0, CNN_Rank_pred ))
	print(CNN_pred, CNN_Rank_pred)


	print('RNN')
	_, RNN_pred, _ = thandler.test( models.RNN(), test_loader, model_name='RNN.pkl' )
	_, RNN_Rank_pred, _ = thandler.test( models.RNN_Rank(), test_loader, model_name='RNN_Rank.pkl' )
	#print(RNN_pred, RNN_Rank_pred)
	#RNN_pred = list(map(lambda m: 1.0 if m > 0.5 else 0.0, RNN_pred ))
	#RNN_Rank_pred = list(map(lambda m: 1.0 if m > 0.5 else 0.0, RNN_Rank_pred ))
	print(RNN_pred, RNN_Rank_pred)
	print('Attention')
	_, ATT_pred, _ = thandler.test( models.Att(), test_loader, model_name='Att.pkl' )
	w1 = models.get_weight()
	_, ATT_Rank_pred, _ = thandler.test( models.Att_Rank(), test_loader, model_name='Att_Rank.pkl' )
	w2 = models.get_weight()
	#print(ATT_pred, ATT_Rank_pred)

	w1 = ([i.tolist() for i in w1.data.cpu()[0][0]])
	w2 = ([i.tolist() for i in w2.data.cpu()[0][0]])
	
	c_ = open(testing).read()
	c_ = util.normalizeString(c_)
	c_ = nltk.word_tokenize(c_)
	for i in zip(c_, w1, w2):
		print(i[0], i[1][0], i[2][0])
	#ATT_pred = list(map(lambda m: 1.0 if m > 0.5 else 0.0, ATT_pred ))
	#ATT_Rank_pred = list(map(lambda m: 1.0 if m > 0.5 else 0.0, ATT_Rank_pred ))
	print(ATT_pred, ATT_Rank_pred)
	
if __name__ == '__main__':
	thandler = training_handler.handler(args.process_command())	
	testing = thandler.testing_data()

	#jsons = os.listdir(training)
	#random.seed(10)
	#random.shuffle(jsons)
	RUN(testing)
	
	#RUN(['{}/{}'.format(training, i) for i in jsons])
