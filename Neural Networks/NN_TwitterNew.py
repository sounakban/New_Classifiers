import numpy as np

#-------------------------------------------Functions-------------------------------------------

def get_Embeddings(data=[], selected_terms = set()):
	import os
	import pickle

	fileName = "saveFiles/Twit_Embeddings.pkl"
	if os.path.exists(fileName):
		with open(fileName, 'rb') as temp:
			doc_vectors, embeddings, maxSize, embedding_vocab = pickle.load(temp)
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
import pandas as pd

dataDF = pd.read_csv('/home/sounak/Resources/Data/Twitter for test/training.1600000.processed.noemoticon.csv', encoding='latin1', header=None)
labels = dataDF[0].tolist()
data = dataDF[5].tolist()
index_shuf = list(range(len(data)))
shuffle(index_shuf)
data = [data[i] for i in index_shuf]
labels = [labels[i] for i in index_shuf]
labels = np.array(labels)

## Binarize Labels ##
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# To make labels 2D instead of 1D default behavior
labels = np.hstack((labels, 1 - labels))
print("Label dimention : ", labels.shape)

from Tools.Feature_Extraction import chisqure
selected_terms = chisqure(data, labels, feature_count = 12000)

## Process Dataset ##
data_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(data, selected_terms)



totrec = 0.0
totprec = 0.0
totF1 = 0.0

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
from Tools.Classifier import CNN_Classifier, CNN2_Classifier, RNN_Classifier, Stacked_BiLSTM_Classifier, Nested_CNN_Classifier

# classifier = CNN_Classifier(filter_sizes=[3,5,7], filter_counts=[500,400,250], pool_windows=[6,5,3], learning_rate=0.0001, batch_size=32, num_epochs=12)
# CNN Clssifier parameters based on Paper
# classifier = CNN2_Classifier(filter_sizes=[3,5,7], filter_counts=[40,40,40], pool_windows=[10,10,10], learning_rate=0.001, batch_size=64, num_epochs=70)
# classifier = RNN_Classifier(output_size=256, learning_rate=0.001, batch_size=64, num_epochs=7)
# classifier = Nested_CNN_Classifier(filter_sizes=[6,2], filter_counts=[300,200], pool_windows=[2,2], learning_rate=0.001, batch_size=64, num_epochs=30)
classifier = Stacked_BiLSTM_Classifier(output_size=50, learning_rate=0.001, batch_size=256, num_epochs=15)

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
