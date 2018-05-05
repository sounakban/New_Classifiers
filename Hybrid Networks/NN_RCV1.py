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
		doc_vectors = pad_sequences(doc_vectors, maxlen=maxSize, padding='post', value=0.)
		train_doc_vectors = doc_vectors[:len(train_docs)]
		test_doc_vectors = doc_vectors[len(train_docs):]

		# with open(fileName, 'wb') as temp:
		# 	pickle.dump((train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab), temp)

	print("Embeddings Shape : ",embeddings.shape)
	return (train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab)




#-------------------------------------------Feature Extraction-------------------------------------------

# #Corpus Data
# from sklearn.datasets import fetch_rcv1
# rcv1_info = fetch_rcv1()
# sklearn_labelMatrix = rcv1_info.target.toarray()
# sklearn_docIDs = rcv1_info.sample_id
# rcv1_info = []
#
# RCV1V2Path = "/home/sounak/Resources/Data/RCV1-V2/Raw Data/"
# from Tools.getRCV1V2 import getRCV1V2
# rcv1v2_data = getRCV1V2(RCV1V2Path, testset=1)
#
#
# # train_docs_index = range(rcv1v2_data.getTrainDocCount())
# # print "Num of train Docs: ", len(train_docs_index)
# # test_docs_index = range(rcv1v2_data.getTrainDocCount(), rcv1v2_data.getTotalDocCount())
# # print "Num of test Docs: ", len(test_docs_index)
# # test_docs_index = [test_docs_index[2]]
#
# documents = rcv1v2_data.getData()
# # train_docs = [documents[doc_ind] for doc_ind in train_docs_index]
# train_docs = documents[:rcv1v2_data.getTrainDocCount()]
# # test_docs = [documents[doc_ind] for doc_ind in test_docs_index]
# test_docs = documents[rcv1v2_data.getTrainDocCount():]
# del documents
#
# #Get Doc-Label Matrices
# # test_docIDs = [rcv1v2_data.getDocIDs()[i] for i in test_docs_index]
# test_docIDs = rcv1v2_data.getDocIDs()[rcv1v2_data.getTrainDocCount():]
# sklearn_testIndices = []
# sklearn_testIndices.append(np.where(sklearn_docIDs == test_docIDs[0])[0][0])
# i = sklearn_testIndices[0]+1; j = 1
# while(j<len(test_docIDs)):
#     if sklearn_docIDs[i] == test_docIDs[j]:
#         sklearn_testIndices.append(i)
#         i+=1; j+=1
#     else:
#         print("Disparity at test doc num: ", j)
#         break
# # train_labels = numpy.array([sklearn_labelMatrix[i] for i in train_docs_index])
# train_labels = sklearn_labelMatrix[:rcv1v2_data.getTrainDocCount(), :]
# test_labels = np.array([sklearn_labelMatrix[i] for i in sklearn_testIndices])


################################################################################################################################

#Corpus Data
from sklearn.datasets import fetch_rcv1
rcv1_info = fetch_rcv1()
sklearn_labelMatrix = rcv1_info.target.toarray()
sklearn_docIDs = rcv1_info.sample_id
rcv1_info = []

from Tools.getRCV1 import getRCV1
from Tools.getRCV1V2 import getRCV1V2
RCV1Path = "/home/sounak/Resources/Data/rcv1_train_test/Data/"
RCV1V2Path = "/home/sounak/Resources/Data/RCV1-V2/Raw Data/"
rcv1_data = getRCV1(RCV1Path, RCV1V2Path, testset=1)
rcv1v2_data = getRCV1V2(RCV1V2Path, testset=1)


train_docs_index = range(rcv1_data.getTrainDocCount())
print("Num of train Docs: ", len(train_docs_index))
test_docs_index = range(rcv1_data.getTrainDocCount(), rcv1_data.getTotalDocCount())
print("Num of test Docs: ", len(test_docs_index))

documents = rcv1_data.getData()
train_docs = [documents[doc_ind] for doc_ind in train_docs_index]
test_docs = [documents[doc_ind] for doc_ind in test_docs_index]
documents = []

#Get Doc-Label Matrices
# test_docIDs = [rcv1v2_data.getDocIDs()[i] for i in test_docs_index]
test_docIDs = rcv1v2_data.getDocIDs()[rcv1v2_data.getTrainDocCount():]
sklearn_testIndices = []
sklearn_testIndices.append(np.where(sklearn_docIDs == test_docIDs[0])[0][0])
i = sklearn_testIndices[0]+1; j = 1
while(j<len(test_docIDs)):
    if sklearn_docIDs[i] == test_docIDs[j]:
        sklearn_testIndices.append(i)
        i+=1; j+=1
    else:
        print("Disparity at test doc num: ", j)
        break
# train_labels = numpy.array([sklearn_labelMatrix[i] for i in train_docs_index])
train_labels = sklearn_labelMatrix[:rcv1v2_data.getTrainDocCount(), :]
test_labels = np.array([sklearn_labelMatrix[i] for i in sklearn_testIndices])

#Unused variables
sklearn_labelMatrix = []; sklearn_testIndices = []; sklearn_docIDs = []; rcv1_data = []

#Clear memory
rcv1_data = []
rcv1v2_data = []



###################################################################################################################################


# -------------------------------------------Prepare Data-------------------------------------------

## Process Dataset ##
from Tools.Feature_Extraction import chisqure
selected_terms = chisqure(train_docs, train_labels, feature_count = 10000)
train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(train_docs, test_docs, selected_terms)


# -------------------------------------------Classification-------------------------------------------

from Tools.Classifier import CNN_Classifier, RNN_Classifier, KerasBlog_CNN_Classifier

# classifier = CNN_Classifier(filter_sizes=[3,7], filter_counts=[150,300], pool_windows=[6,21], learning_rate=0.001, batch_size=32, num_epochs=100)
# classifier = RNN_Classifier(output_size=256, learning_rate=0.001, batch_size=7, num_epochs=100)
classifier = KerasBlog_CNN_Classifier(filter_sizes=[5,5], filter_counts=[600,600], pool_windows=[2,2], learning_rate=0.001, batch_size=128, num_epochs=15)
print("50 epochs with random initialization of embeddings.")

# Multi-class Clasification
# new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])
# Multi-label Clasification
new = classifier.predict_multilabel(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])
