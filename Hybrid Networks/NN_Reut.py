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
		embeddingGenerator = Get_Embeddings()
		doc_vectors, embeddings, maxSize, embedding_vocab = embeddingGenerator.googleVecs(all_docs, selected_terms)
		del embeddingGenerator
		from keras.preprocessing.sequence import pad_sequences
		doc_vectors = pad_sequences(doc_vectors, maxlen=maxSize, padding='post', value=0.)
		# from Tools.Utils import pad_sequences3D
		# doc_vectors = pad_sequences3D(doc_vectors, maxSize[0], maxSize[1], value=0)
		train_doc_vectors = doc_vectors[:len(train_docs)]
		test_doc_vectors = doc_vectors[len(train_docs):]

		# with open(fileName, 'wb') as temp:
		# 	pickle.dump((train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab), temp)

	print("Embeddings Shape : ",embeddings.shape)
	return (train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab)



## List of document ids ##

def getDocIDs_top10():
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


def getDocIDs_90():
	# 90 Categories
	documents = reuters.fileids()
	train_docs_id = list(filter(lambda doc: doc.startswith("train") and len(reuters.raw(doc))>51, documents))
	test_docs_id = list(filter(lambda doc: doc.startswith("test") and len(reuters.raw(doc))>51, documents))
	return (train_docs_id, test_docs_id)




#-------------------------------------------Prepare Data-------------------------------------------

# dataset = "Top10"
# train_docs_id, test_docs_id = getDocIDs_top10()
dataset = "All90"
train_docs_id, test_docs_id = getDocIDs_90()

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]

# Transform multilabel labels ##
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
print("Label dimention : ", train_labels.shape)
test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])

## Process Dataset ##
from Tools import Utils
train_docs, test_docs = Utils.preprocess(train_docs, test_docs)
from Tools.Feature_Extraction import chisqure
selected_terms = chisqure(train_docs, train_labels, feature_count = 500)
# print(len(train_docs), " ; ", len(test_docs))
train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(dataset, train_docs, test_docs, selected_terms)
# print("Doc Vector : ", train_doc_vectors[10])



#-------------------------------------------Classification-------------------------------------------

from Tools.Classifier import HNN_RR_Classifier, HNN_CR_Classifier, HNN_RC_Classifier, KerasBlog_CNN_Classifier

# classifier = HNN_RC_Classifier(RNN_output_size=256, filter_sizes=[3,7], filter_counts=[150,300], pool_windows=[6,21], learning_rate=0.001, batch_size=7, num_epochs=100)
# new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize[0], maxSize[1], train_labels.shape[1])


classifier = KerasBlog_CNN_Classifier(filter_sizes=[5,5], filter_counts=[300,300], pool_windows=[2,2], learning_rate=0.001, batch_size=128, num_epochs=17)
# Multi-class Clasification
# new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])
# Multi-label Clasification
new = classifier.predict_multilabel(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])
