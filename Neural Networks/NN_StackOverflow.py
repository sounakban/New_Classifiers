import numpy as np

#-------------------------------------------Functions-------------------------------------------

def get_Embeddings(data=[], selected_terms = set()):
	import os
	import pickle

	fileName = "saveFiles/StkOvf_Embeddings.pkl"
	if os.path.exists(fileName):
		with open(fileName, 'rb') as temp:
			data_vectors, embeddings, maxSize, embedding_vocab = pickle.load(temp)
	else:
		all_docs = list(data)

		# Get Embeddings
		from Tools.Load_Embedings import Get_Embeddings
		embeddingGenerator = Get_Embeddings()
		data_vectors, embeddings, maxSize, embedding_vocab = embeddingGenerator.googleVecs(all_docs, selected_terms)
		del embeddingGenerator
		from keras.preprocessing.sequence import pad_sequences
		data_vectors = pad_sequences(data_vectors, maxlen=maxSize, padding='post', value=0.)

		# with open(fileName, 'wb') as temp:
		# 	pickle.dump((data_vectors, embeddings, maxSize, embedding_vocab), temp)

	print("Embeddings Shape : ",embeddings.shape)
	return (data_vectors, embeddings, maxSize, embedding_vocab)


#-------------------------------------------Prepare Data-------------------------------------------

from Tools.getStackOverflow import getStackOverflow
from random import sample

# SOvrflow = getStackOverflow("/Volumes/Files/Work/Research/Information Retrieval/1) Data/StackOverflow-Dataset/")
SOvrflow = getStackOverflow("/home/sounak/Datasets/StackOverflow-Dataset/")
data = SOvrflow.getData()
labels = SOvrflow.getTarget()

## Binarize Labels ##
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print("Label dimention : ", labels.shape)

from Tools.Feature_Extraction import chisqure
selected_terms = chisqure(data, labels, feature_count = 1500)

## Process Dataset ##
data_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(data, selected_terms)


#-------------------------------------------Classification-------------------------------------------

totrec = 0.0
totprec = 0.0
totF1 = 0.0

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
from Tools.Classifier import CNN1_Classifier, RNN_Classifier, Nested_CNN_Classifier

# classifier = CNN_Classifier(filter_sizes=[5,7,9], filter_counts=[500,350,250], pool_windows=[6,4,3], learning_rate=0.001, batch_size=64, num_epochs=7)
# classifier = Nested_CNN_Classifier(filter_sizes=[6,2], filter_counts=[300,150], pool_windows=[2,2], learning_rate=0.001, batch_size=64, num_epochs=7)
# classifier = RNN_Classifier(output_size=512, learning_rate=0.001, batch_size=7, num_epochs=100)
classifier = CNN1_Classifier(filter_sizes=[3,5,7], filter_counts=[100,100,100], pool_windows=[5,6,7], learning_rate=0.001, batch_size=50, num_epochs=41)

for train_indices, test_indices in kf.split(data_vectors):
	train_doc_vectors, train_labels = [data_vectors[i] for i in train_indices], labels[train_indices]  #[labels[i] for i in train_indices]
	test_doc_vectors, test_labels = [data_vectors[i] for i in test_indices], labels[test_indices]  #[labels[i] for i in test_indices]

	new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])


#     from sklearn.metrics import f1_score, precision_score, recall_score
#
#     #MICRO
#     precision = precision_score(test_labels, predictions, average='micro')
#     totprec += precision
#     recall = recall_score(test_labels, predictions, average='micro')
#     totrec += recall
#     f1 = f1_score(test_labels, predictions, average='micro')
#     totF1 += f1
#
#     print("Micro-average quality numbers")
#     print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
#             .format(precision, recall, f1))
#
#     #MACRO
#     precision = precision_score(test_labels, predictions, average='macro')
#     recall = recall_score(test_labels, predictions, average='macro')
#     f1 = f1_score(test_labels, predictions, average='macro')
#
#     print("Macro-average quality numbers")
#     print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
#             .format(precision, recall, f1))
#
#     #INDIVIDUAL
#     precision = precision_score(test_labels, predictions, average=None)
#     recall = recall_score(test_labels, predictions, average=None)
#     f1 = f1_score(test_labels, predictions, average=None)
#
#     print("All-Class quality numbers")
#     print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
#             .format(precision, recall, f1))
#
# print "10-fold Micro average:"
# print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
#         .format(totprec/K, totrec/K, totF1/K))
#
#
#
# # Transform multilabel labels
# train_labels = [(labels[i],) for i in train_indices]
# test_labels = [(labels[i],) for i in test_indices]
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
