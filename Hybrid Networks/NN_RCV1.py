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

#Corpus Data
from sklearn.datasets import fetch_rcv1
rcv1_info = fetch_rcv1()
sklearn_labelMatrix = rcv1_info.target.toarray()
sklearn_docIDs = rcv1_info.sample_id
rcv1_info = []

RCV1V2Path = "/home/sounak/Resources/Data/RCV1-V2/Raw Data/"
from Tools.getRCV1V2 import getRCV1V2
rcv1v2_data = getRCV1V2(RCV1V2Path, testset=1)


# train_docs_index = range(rcv1v2_data.getTrainDocCount())
# print "Num of train Docs: ", len(train_docs_index)
# test_docs_index = range(rcv1v2_data.getTrainDocCount(), rcv1v2_data.getTotalDocCount())
# print "Num of test Docs: ", len(test_docs_index)
#test_docs_index = [test_docs_index[2]]

documents = rcv1v2_data.getData()
# train_docs = [documents[doc_ind] for doc_ind in train_docs_index]
train_docs = documents[:rcv1v2_data.getTrainDocCount()]
# test_docs = [documents[doc_ind] for doc_ind in test_docs_index]
test_docs = documents[rcv1v2_data.getTrainDocCount():]
del documents

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






# -------------------------------------------Prepare Data-------------------------------------------

## Process Dataset ##
from Tools.Feature_Extraction import chisqure
selected_terms = chisqure(train_docs, train_labels, feature_count = 5000)
train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(train_docs, test_docs, selected_terms)


# -------------------------------------------Classification-------------------------------------------

from Tools.Classifier import CNN_Classifier, RNN_Classifier, KerasBlog_CNN_Classifier

# classifier = CNN_Classifier(filter_sizes=[3,7], filter_counts=[150,300], pool_windows=[6,21], learning_rate=0.001, batch_size=32, num_epochs=100)
# classifier = RNN_Classifier(output_size=256, learning_rate=0.001, batch_size=7, num_epochs=100)
classifier = KerasBlog_CNN_Classifier(filter_sizes=[5,5,5], filter_counts=[200,200,200], pool_windows=[5,5,5], learning_rate=0.001, batch_size=128, num_epochs=12)

new = classifier.predict(np.array(train_doc_vectors), train_labels, np.array(test_doc_vectors), test_labels, embeddings, maxSize, train_labels.shape[1])



# #-------------------------- Evaluation ----------------------
# precision = precision_score(test_labels, predictions, average='micro')
# recall = recall_score(test_labels, predictions, average='micro')
# f1 = f1_score(test_labels, predictions, average='micro')
#
# print("Micro-average quality numbers")
# print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
#
# precision = precision_score(test_labels, predictions, average='macro')
# recall = recall_score(test_labels, predictions, average='macro')
# f1 = f1_score(test_labels, predictions, average='macro')
#
# print("Macro-average quality numbers")
# print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
#
#
# #print [len(numpy.nonzero(train_labels[:, i])[0].tolist()) for i in range(num_labels)]
# #print [len(numpy.nonzero(test_labels[:, i])[0].tolist()) for i in range(num_labels)]
# precision = precision_score(test_labels, predictions, average=None)
# recall = recall_score(test_labels, predictions, average=None)
# f1 = f1_score(test_labels, predictions, average=None)
# print f1
#
# print "Evaluation complete and it took : ", print_time(start_time)
#
#
# import numpy as np
# exp_train = np.sum(train_labels, axis=0)
# exp_test = np.sum(test_labels, axis=0)
# print "Num of train docs per category:\n", exp_train
# print "Num of test docs per category:\n", exp_test
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
#
#
#
#
# print "Total time taken : ", (time.time() - program_start)/60.0, "minuites"
