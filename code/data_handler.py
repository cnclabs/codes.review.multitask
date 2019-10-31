import json
import numpy as np
import random
from itertools import combinations, permutations

wv_model = None

def set_embedding(embedding):
	global wv_model
	wv_model=embedding

def loadJSON( JSON ):
	_ = json.load(JSON)

	#category
	#review_count
	#avg_rating_stars#
	#rating_stats
	#entity
	#reviews

	auxs = map(dict, _['spoiler_reviews'])
	aux, aux_rating = zip(*[(i['content'], i['rating']) for i in auxs])

	content = wv_model.sentence2vec(_['description'])
	y = _['rating']

	return content, y, aux, aux_rating

def load_data( jsons ):
	contents, ys, auxs, aux_ratings = zip(*list(map( loadJSON, map(open, jsons) )))

	return contents, ys

def rankNet_data( jsons, sample_size ):
	contents = []
	aux1 = []
	aux2 = []
	ys = []
	labels = []

	for i in jsons:
		content, y, aux, aux_rating  = loadJSON(open(i))
		
		aux_pair_org = list(zip(aux, aux_rating))
		aux_pair_rnd = list(zip(aux, aux_rating))
		aux_pair_rnd.reverse()
		#np.random.shuffle(aux_pair_rnd)
		
		for j in zip(aux_pair_org[:sample_size], aux_pair_rnd[:sample_size]):
			aux1.append(wv_model.sentence2vec(j[0][0]))
			aux2.append(wv_model.sentence2vec(j[1][0]))
			label = 0.0 if j[0][1]-j[1][1] < 0 else 1.0
			labels.append(label)
			ys.append(y)
			contents.append(content)

		'''
		
		aux_pair_org = list(zip(aux[:ssize], aux_rating[:ssize]))
		aux_pair = list(combinations(aux_pair_org, 2))
		plength = len(aux_pair)

		for j in aux_pair:
			aux1.append(sentence2vec(j[0][0]))
			aux2.append(sentence2vec(j[1][0]))

			label = 0.0 if j[0][1]-j[1][1] < 0 else 1.0
			labels.append(label)
			ys.append(y)
			contents.append(content)
		'''

	return contents, aux1, aux2, ys, labels

def regression_data( jsons, sample_size ):
	contents = []
	aux1 = []
	ys = []
	labels = []

	for i in jsons:
		content, y, aux, aux_rating  = loadJSON(open(i))
		
		aux_pair_org = list(zip(aux, aux_rating))
		
		for j in aux_pair_org[:sample_size]:
			aux1.append(wv_model.sentence2vec(j[0]))
			labels.append(j[1])
			ys.append(y)
			contents.append(content)

	return contents, aux1, ys, labels

def testing_data( data ):
	content = wv_model.sentence2vec(open(data).read())
		
	return [contents], [0]
