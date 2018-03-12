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
from Tools.Classifier import CNN_Classifier, RNN_Classifier, DNN_Classifier

# classifier = DNN_Classifier(learning_rate=0.001, batch_size=32, num_epochs=30)
classifier = CNN_Classifier(filter_sizes=[3,7], filter_counts=[300,250], pool_windows=[6,3], learning_rate=0.001, batch_size=32, num_epochs=30)
# classifier = CNN_Classifier(filter_sizes=[5], filter_counts=[200], pool_windows=[2], learning_rate=0.001, batch_size=32, num_epochs=30)
# classifier = RNN_Classifier(output_size=256, learning_rate=0.001, batch_size=7, num_epochs=100)

for train_indices, test_indices in kf.split(data_vectors):
	train_doc_vectors, train_labels = [data_vectors[i] for i in train_indices], labels[train_indices]  #[labels[i] for i in train_indices]
	test_doc_vectors, test_labels = [data_vectors[i] for i in test_indices], labels[test_indices]  #[labels[i] for i in test_indices]

	new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])




# for i in range(K):
#     """
#     pos_train_indeces = sample(range(len(postweet)), int(len(postweet)*train_cut) )
#     neg_train_indeces = sample(range(len(negtweet)), int(len(negtweet)*train_cut) )
#     pos_test_indeces = list( set(range(len(postweet))) - set(pos_train_indeces) )
#     neg_test_indeces = list( set(range(len(postweet))) - set(pos_train_indeces) )
#     """
#
#     pos_test_indeces = range(i*pos_set_size, (i+1)*pos_set_size)
#     neg_test_indeces = range(i*neg_set_size, (i+1)*neg_set_size)
#     pos_train_indeces = list( set(range(len(postweet))) - set(pos_test_indeces) )
#     neg_train_indeces = list( set(range(len(postweet))) - set(neg_test_indeces) )
#     print len(pos_test_indeces), len(neg_test_indeces), len(pos_train_indeces), len(neg_train_indeces)
#
#     train_docs = [postweet[i] for i in pos_train_indeces]
#     train_docs.extend( [negtweet[i] for i in neg_train_indeces] )
#     test_docs = [postweet[i] for i in pos_test_indeces]
#     test_docs.extend( [negtweet[i] for i in neg_test_indeces] )
#
#
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.preprocessing import MultiLabelBinarizer
#     stop_words = cachedStopWords
#
#
#     # Learn and transform train documents
#     # Tokenisation
#     vectorizer = TfidfVectorizer(stop_words=stop_words,
#                                  tokenizer=tokenize)
#     vectorised_train_documents = vectorizer.fit_transform(train_docs)
#     print vectorised_train_documents.shape
#     vectorised_test_documents = vectorizer.transform(test_docs)
#
#
#     test_labels = ["pos"] * pos_set_size
#     test_labels.extend(["neg"] * neg_set_size)
#     train_labels = ["pos"] * (len(postweet) - pos_set_size)
#     train_labels.extend(["neg"] * (len(negtweet) - neg_set_size))
#
#
#
#
#     #-------------------------------------------Classification-------------------------------------------
#     from sklearn.multiclass import OneVsRestClassifier
#
#     """
#     from sklearn.naive_bayes import GaussianNB
#     classifier = OneVsRestClassifier(GaussianNB())
#     #"""
#     """
#     from sklearn.svm import LinearSVC
#     classifier = OneVsRestClassifier(LinearSVC(random_state=42))
#     #"""
#     """
#     from sklearn.naive_bayes import MultinomialNB
#     #classifier = OneVsRestClassifier(MultinomialNB(alpha=0.01))
#     classifier = OneVsRestClassifier(MultinomialNB())
#     #"""
#     #"""
#     from sklearn.neighbors import KNeighborsClassifier
#     classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=100, n_jobs=-2))
#     #"""
#
#
#     classifier.fit(vectorised_train_documents, train_labels)
#
#     predictions = classifier.predict(vectorised_test_documents)
#
#
#     #-------------------------------------------Evaluation-------------------------------------------
#
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
#         .format(totprec/10, totrec/10, totF1/10))
#
#
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
