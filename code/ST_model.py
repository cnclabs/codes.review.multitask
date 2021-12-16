import torch
import torch.nn as nn
import torch.nn.functional as F
import args
from transformer import MultiHeadAtt as attention
from transformers import BertModel

embed_size = 300
weight = 'bert-base-cased'
if args.process_command().lang == 'jp':
	weight = 'cl-tohoku/bert-base-japanese-whole-word-masking'

class BERT(nn.Module):
	
	def __init__(self, bert_model = BertModel, bert_weight = weight, classes=2):
		super(BERT, self).__init__()
		self.loss = nn.MSELoss()
		self.pretrained = bert_model.from_pretrained(bert_weight)

		for param in self.pretrained.parameters():
			param.requires_grad = False
		for param in self.pretrained.encoder.layer[-1].parameters():
			param.requires_grad = True
		for param in self.pretrained.encoder.layer[-2].parameters():
			param.requires_grad = True

		self.dropout = nn.Dropout(p=0.1)
		self.final = nn.Linear(768, 1)

	def loss_func(self):
		return self.loss

	def main_task(self, data):
		ratings = [] 
		for d in data:
			token_ids, attn_mask, seg_ids = d
			token_ids = token_ids.view(1,302)
			attn_mask = attn_mask.view(1,302)
			seg_ids = seg_ids.view(1,302)
			output = self.pretrained(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
			hidden_reps, cls_head = output.last_hidden_state, output.pooler_output
			cls_head = self.dropout(cls_head)
			rating = self.final(cls_head)
			ratings.append(rating)

		ratings = torch.cat(ratings, dim=1)

		return cls_head, ratings[0]

	def forward(self, data , mode='train'):
		cls_head, y_rating = self.main_task(data[2])

		if mode == 'train':
			return self.loss(y_rating, data[1])
		else:
			return y_rating.view(y_rating.size(0),)

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.loss = nn.MSELoss()
		
		self.doc0 = nn.Linear(300, 10)
		self.doc1 = nn.Linear(10, 10)
		self.final = nn.Linear(10, 1)

	def loss_func(self):
		return self.loss

	def main_task(self, x):
		x = F.avg_pool2d(x, (x.size(1), 1)).squeeze(1)
		x = self.doc0(x)
		x = F.relu(x)
		
		x1 = self.doc1(x)
		x1 = F.relu(x1)
		rating = self.final(x1)

		return x1, rating

	def forward(self, data , mode='train'):
		doc, y_rating = self.main_task(data[0])

		if mode == 'train':
			return self.loss(y_rating, data[1])
		else:
			return y_rating.view(y_rating.size(0),)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.loss = nn.MSELoss()
		
		self.conv0 = nn.Conv2d(1, 10, (3, 300), padding=(1, 0))
		self.final = nn.Linear(10, 1)

	def loss_func(self):
		return self.loss

	def main_task(self, x):
		x = F.relu(self.conv0(x)).squeeze(3)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		
		rating = self.final(x)

		return x, rating

	def forward(self, data, mode='train'):
		x = data[0].unsqueeze(1)
		doc, y_rating = self.main_task(x)

		if mode == 'train':
			return self.loss(y_rating, data[1])
		else:
			return y_rating.view(y_rating.size(0),)

class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()
		self.loss = nn.MSELoss()
		
		self.rnn = nn.GRU(embed_size, embed_size, 1, bidirectional=True)
		self.final = nn.Linear(embed_size*2, 1)

	def loss_func(self):
		return self.loss

	def main_task(self, x):
		x, hidden = self.rnn(x)
		x = F.avg_pool2d(x, (x.size(1), 1)).squeeze(1)
		
		rating = self.final(x)

		return x, rating

	def forward(self, x_ , mode='train'):
		x = x_[0]
		doc, y_rating = self.main_task(x)

		if mode == 'train':
			return self.loss(y_rating, x_[1])
		else:
			return y_rating.view(y_rating.size(0),)

class Att(nn.Module):
	def __init__(self):
		super(Att, self).__init__()
		self.loss = nn.MSELoss()
		
		self.att = attention(2, embed_size, 2)
		self.final = nn.Linear(embed_size, 1)

	def loss_func(self):
		return self.loss

	def main_task(self, x):
		x = self.att(x)
		x = F.avg_pool2d(x, (x.size(1), 1)).squeeze(1)
		
		rating = self.final(x)

		return x, rating

	def forward(self, x_ , mode='train'):
		x = x_[0]
		doc, y_rating = self.main_task(x)

		if mode == 'train':
			return self.loss(y_rating, x_[1])
		else:
			return y_rating.view(y_rating.size(0),)
