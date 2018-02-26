#Contains routines for data manipulation s that are different from other modules
import numpy as np

def prob2Onehot(prob_dist):
	ret_vec = []
	for probs in prob_dist:
		probs_list = list(probs)
		temp = [0]*len(probs_list)
		temp[probs_list.index(max(probs_list))] = 1
		ret_vec.append(temp)
	return np.array(ret_vec)

def prob2Onehot3D(prob_dist):
	ret_vec = []
	for doc in prob_dist:
		doc_vec = []
		for probs in doc:
			probs_list = list(probs)
			temp = [0]*len(probs_list)
			temp[probs_list.index(max(probs_list))] = 1
			doc_vec.append(temp)
		ret_vec.append(doc_vec)
	return np.array(ret_vec)


def pad_sequences_3D(vectors, maxlen, value):
	for i in range(len(vectors)):
		while len(vectors[i]) < maxlen:
			vectors[i].append(value)
	return np.array(vectors)
