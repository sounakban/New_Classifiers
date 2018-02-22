import numpy as np
from nltk.corpus import reuters

#-------------------------------------------Functions-------------------------------------------

def get_Embeddings(dataset, train_docs=[], test_docs=[], selected_terms = set()):
	import os
	import pickle

	if dataset == "Top10":
		fileName = "saveFiles/Reut10_Embeddings.pkl"
	else:
		fileName = "saveFiles/Reut90_Embeddings.pkl"

	if os.path.exists(fileName):
		with open(fileName, 'rb') as temp:
			train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = pickle.load(temp)
	else:
		all_docs = list(train_docs)	#Using list function to create deep copy of the object
		all_docs.extend(test_docs)

		# Get Embeddings
		from Tools.Load_Embedings import Get_Embeddings
		from keras.preprocessing.sequence import pad_sequences
		embeddingGenerator = Get_Embeddings()
		doc_vectors, embeddings, maxSize, embedding_vocab = embeddingGenerator.googleVecs(all_docs, selected_terms)
		del embeddingGenerator
		doc_vectors = pad_sequences(doc_vectors, maxlen=maxSize, padding='post', value=0.)
		train_doc_vectors = doc_vectors[:len(train_docs)]
		test_doc_vectors = doc_vectors[len(train_docs):]

		with open(fileName, 'wb') as temp:
			pickle.dump((train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab), temp)

	print("Embeddings Shape : ",embeddings.shape)
	return (train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab)



## List of document ids ##

def getDocIDs_90():
	# Top 10 Categories
	documents = [f for f in reuters.fileids() if len(reuters.categories(fileids=f))==1]
	train_docs_id = list(filter(lambda doc: doc.startswith("train") and len(reuters.raw(doc))>51, documents))
	test_docs_id = list(filter(lambda doc: doc.startswith("test") and len(reuters.raw(doc))>51, documents))
	new_train_docs_id = []
	new_test_docs_id = []
	for cat in reuters.categories():
	  li=[f for f in reuters.fileids(categories=cat) if f in train_docs_id]
	  li_te = [f for f in reuters.fileids(categories=cat) if f in test_docs_id]
	  if len(li)>20 and len(li_te)>20:
		  new_train_docs_id.extend(li)
		  new_test_docs_id.extend(li_te)
	train_docs_id = new_train_docs_id
	test_docs_id = new_test_docs_id
	return (train_docs_id, test_docs_id)


def getDocIDs_top10():
	# 90 Categories
	documents = reuters.fileids()
	train_docs_id = list(filter(lambda doc: doc.startswith("train") and len(reuters.raw(doc))>51, documents))
	test_docs_id = list(filter(lambda doc: doc.startswith("test") and len(reuters.raw(doc))>51, documents))
	return (train_docs_id, test_docs_id)




#-------------------------------------------Prepare Data-------------------------------------------

dataset = "Top10"
train_docs_id, test_docs_id = getDocIDs_top10()
# dataset = "All90"
# train_docs_id, test_docs_id = getDocIDs_90()

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

# Transform multilabel labels ##
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])

## Process Dataset ##
from Tools.Feature_Extraction import chisqure
selected_terms = chisqure(train_docs, train_labels, feature_count = 0)
# print(len(train_docs), " ; ", len(test_docs))
train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(dataset, train_docs, test_docs, selected_terms)



#-------------------------------------------Classification-------------------------------------------

from Tools.Classifier import CNN_Classifier

# classifier = CNN_Classifier(filter_sizes=[3,5,7], filter_counts=[150,200,300], pool_windows=[12,15,14], learning_rate=0.01, batch_size=64, num_epochs=10)
classifier = CNN_Classifier(filter_sizes=[3,7], filter_counts=[150,300], pool_windows=[6,21], learning_rate=0.001, batch_size=32, num_epochs=50)
new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])



# from sklearn.multiclass import OneVsRestClassifier
#
# """
# from sklearn.naive_bayes import BernoulliNB
# classifier = OneVsRestClassifier(BernoulliNB(alpha=0.01))
# #"""
# """
# from sklearn.naive_bayes import MultinomialNB
# classifier = OneVsRestClassifier(MultinomialNB(alpha=0.01))
# #"""
# #"""
# from sklearn.naive_bayes import MultinomialNB
# classifier = OneVsRestClassifier(MultinomialNB())
# #"""
# """
# from sklearn.neighbors import KNeighborsClassifier
# classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=15, n_jobs=-2))
# #"""
# """
# from sklearn.svm import LinearSVC
# classifier = OneVsRestClassifier(LinearSVC(random_state=42))
# #"""
#
#
# classifier.fit(vectorised_train_documents, train_labels)
# predictions = classifier.predict(vectorised_test_documents)
#
#
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
