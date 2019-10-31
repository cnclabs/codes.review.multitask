import data_handler as dh
import math
import random
import os
from collections import defaultdict, deque
from itertools import product
from scipy import spatial
from sklearn.metrics import mean_squared_error as mse

def cosine_similarity( lst1, lst2 ):
	return 1-spatial.distance.cosine(lst1, lst2) 

#url = '../dataset/movie_review'
#url = '../dataset/yelp_round9'
#url = '/tmp2/zllin/kkstream/dataset/imdb'
url = '/tmp2/zllin/kkstream/dataset/filmark/'
k = [10, 20, 30, 40, 50]

data = os.listdir(url)
random.seed(10)
random.shuffle(data)
x, y = dh.preprocess(['{}/{}'.format(url, i) for i in data])

N = len(x)
indices = list(range(N))
split = int(math.floor(0.1*N))
train_idx, valid_idx, test_idx = indices[split*2], indices[split:split*2], indices[:split]

_ = product(test_idx, range(len(x)))

result = defaultdict(dict)


print('---\t cosine similarty\t---')
for i in _:
	result[i[0]].update({i[1]: cosine_similarity(sum(x[i[0]]), sum(x[i[1]]))})


for k_ in k:
	testing_y = deque()
	predicted_y = deque()


	for i in result.keys():
		cs = result[i]
		sorted_cs = sorted(cs.items(), key=lambda kv:kv[1], reverse=True)[1:]

		topK = [j[0] for j in sorted_cs[:k_]]
	
		testing_y.append(y[i])
		predicted_y.append(float(sum([y[_]for _ in topK]))/k_)


	print(('{}:\t{}').format(k_, mse(testing_y, predicted_y) ** 0.5))

