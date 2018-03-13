import numpy as np

#-------------------------------------------Functions-------------------------------------------

def get_Embeddings(data=[], selected_terms = set()):
	import os
	import pickle

	fileName = "saveFiles/Twit_Embeddings.pkl"
	if os.path.exists(fileName):
		with open(fileName, 'rb') as temp:
			doc_vectors, embeddings, maxSize, embedding_vocab = pickle.load(temp)
			# print("Doc vec", doc_vectors[0])
			# print("Embedding vec", embeddings[0])
			# print("vocab", embedding_vocab[0:3])
	else:
		all_docs = list(data)

		# Get Embeddings
		from Tools.Load_Embedings import Get_Embeddings
		embeddingGenerator = Get_Embeddings()
		doc_vectors, embeddings, maxSize, embedding_vocab = embeddingGenerator.googleVecs(all_docs, selected_terms)
		del embeddingGenerator
		from keras.preprocessing.sequence import pad_sequences
		doc_vectors = pad_sequences(doc_vectors, maxlen=maxSize, padding='post', value=0.)

		# with open(fileName, 'wb') as temp:
		# 	pickle.dump((doc_vectors, embeddings, maxSize, embedding_vocab), temp)

	print("Embeddings Shape : ",embeddings.shape)
	return (doc_vectors, embeddings, maxSize, embedding_vocab)





#-------------------------------------------Prepare Data-------------------------------------------

from nltk.corpus import twitter_samples as tweet
# from random import sample
from random import shuffle

postweet = tweet.strings('positive_tweets.json')
negtweet = tweet.strings('negative_tweets.json')
data = list(postweet)
data.extend(negtweet)
labels = [[1,0]]*len(postweet)
labels.extend([[0,1]]*len(negtweet))
index_shuf = list(range(len(data)))
shuffle(index_shuf)
data = [data[i] for i in index_shuf]
labels = [labels[i] for i in index_shuf]
labels = np.array(labels)

from Tools.Feature_Extraction import chisqure
selected_terms = chisqure(data, labels, feature_count = 800)

## Process Dataset ##
data_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(data, selected_terms)



totrec = 0.0
totprec = 0.0
totF1 = 0.0

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
from Tools.Classifier import CNN_Classifier, RNN_Classifier, BDRNN_Classifier

# classifier = CNN_Classifier(filter_sizes=[3,7], filter_counts=[400, 250], pool_windows=[6,3], learning_rate=0.001, batch_size=64, num_epochs=15) #Best epoch : 10
classifier = RNN_Classifier(output_size=256, learning_rate=0.001, batch_size=64, num_epochs=12)
# classifier = BDRNN_Classifier(output_size=256, learning_rate=0.001, batch_size=256, num_epochs=15)

for train_indices, test_indices in kf.split(data_vectors):
	train_doc_vectors, train_labels = [data_vectors[i] for i in train_indices], labels[train_indices]  #[labels[i] for i in train_indices]
	test_doc_vectors, test_labels = [data_vectors[i] for i in test_indices], labels[test_indices]  #[labels[i] for i in test_indices]

	new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])



# # Transform multilabel labels
# train_labels = [(k,) for k in train_labels]
# test_labels = [(k,) for k in test_labels]
# mlb = MultiLabelBinarizer()
# train_labels = mlb.fit_transform(train_labels)
# test_labels = mlb.transform(test_labels)
# import numpy as np
# exp_train = np.sum(train_labels, axis=0)
# exp_test = np.sum(test_labels, axis=0)
# print "Num of train docs per category:\n", exp_train
# print "Num of test docs per category:\n", exp_test
#
#
# #Export to Spreadsheet
# import xlsxwriter
#
# export = np.column_stack((exp_train, exp_test, f1, precision, recall))
# workbook = xlsxwriter.Workbook('classscores.xlsx')
# worksheet = workbook.add_worksheet()
# row = 0
# for (x,y), value in np.ndenumerate(export):
#     worksheet.write(x, y, value)
# workbook.close()
