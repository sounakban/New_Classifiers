import numpy as np

#-------------------------------------------Functions-------------------------------------------

def get_Embeddings(train_docs=[], test_docs=[], selected_terms = set()):
	import os
	import pickle

	fileName = "saveFiles/20NG_Embeddings.pkl"
	if os.path.exists(fileName):
		with open(fileName, 'rb') as temp:
			train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = pickle.load(temp)
	else:
		all_docs = list(train_docs)
		all_docs.extend(test_docs)

		# Get Embeddings
		from Tools.Load_Embedings import Get_Embeddings
		embeddingGenerator = Get_Embeddings()
		doc_vectors, embeddings, maxSize, embedding_vocab = embeddingGenerator.googleVecs(all_docs, selected_terms)
		del embeddingGenerator
		from keras.preprocessing.sequence import pad_sequences
		maxSize = 1000
		doc_vectors = pad_sequences(doc_vectors, maxlen=maxSize, padding='post', value=0.)
		train_doc_vectors = doc_vectors[:len(train_docs)]
		test_doc_vectors = doc_vectors[len(train_docs):]

		# with open(fileName, 'wb') as temp:
		# 	pickle.dump((train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab), temp)

	print("Embeddings Shape : ",embeddings.shape)
	return (train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab)







#-------------------------------------------Prepare Data-------------------------------------------
## Get Dataset ##
from sklearn.datasets import fetch_20newsgroups

train_docs = fetch_20newsgroups(subset='train').data
train_labels = fetch_20newsgroups(subset='train').target
test_docs = fetch_20newsgroups(subset='test').data
test_labels = fetch_20newsgroups(subset='test').target
print("Total Doc Count : ", len(train_docs)+len(test_docs))

## Binarize Labels ##
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
print("Label dimention : ", train_labels.shape)
test_labels = lb.transform(test_labels)

## Process Dataset ##
from Tools import Utils
train_docs, test_docs = Utils.preprocess(train_docs, test_docs)
from Tools.Feature_Extraction import chisqure
selected_terms = chisqure(train_docs, train_labels, feature_count = 5000)
# print(len(train_docs), " ; ", len(test_docs))
train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(train_docs, test_docs, selected_terms)



# #-------------------------------------------Classification-------------------------------------------

from Tools.Classifier import CNN_Classifier, RNN_Classifier, KerasBlog_CNN_Classifier

# classifier = CNN_Classifier(filter_sizes=[3,7], filter_counts=[150,300], pool_windows=[6,21], learning_rate=0.001, batch_size=32, num_epochs=100)
# classifier = RNN_Classifier(output_size=256, learning_rate=0.001, batch_size=7, num_epochs=100)
classifier = KerasBlog_CNN_Classifier(filter_sizes=[5,5,5], filter_counts=[400,400,400], pool_windows=[5,5,5], learning_rate=0.001, batch_size=128, num_epochs=50)

new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])



# #-------------------------------------------Evaluation-------------------------------------------
#
# from sklearn.metrics import f1_score, precision_score, recall_score
#
# #MICRO
# precision = precision_score(test_labels, predictions,
#                             average='micro')
# recall = recall_score(test_labels, predictions,
#                       average='micro')
# f1 = f1_score(test_labels, predictions, average='micro')
#
# print("Micro-average quality numbers")
# print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
#         .format(precision, recall, f1))
#
# #MACRO
# precision = precision_score(test_labels, predictions,
#                             average='macro')
# recall = recall_score(test_labels, predictions,
#                       average='macro')
# f1 = f1_score(test_labels, predictions, average='macro')
#
# print("Macro-average quality numbers")
# print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
#         .format(precision, recall, f1))
#
# #INDIVIDUAL
# precision = precision_score(test_labels, predictions,
#                             average=None)
# recall = recall_score(test_labels, predictions,
#                       average=None)
# f1 = f1_score(test_labels, predictions, average=None)
#
# print("All-Class quality numbers")
# print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
#         .format(precision, recall, f1))
#
#
# # Transform multilabel labels
# mlb = MultiLabelBinarizer()
# train_labels = mlb.fit_transform([[v] for v in fetch_20newsgroups(subset='train').target.tolist()])
# test_labels = mlb.transform([[v] for v in fetch_20newsgroups(subset='test').target.tolist()])
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
# worksheet.write(0, 0, "Train(Count)")
# worksheet.write(0, 1, "Test(Count)")
# worksheet.write(0, 2, "F1")
# worksheet.write(0, 3, "Precision")
# worksheet.write(0, 4, "Recall")
# for (x,y), value in np.ndenumerate(export):
#     worksheet.write(x+1, y, value)
# workbook.close()
