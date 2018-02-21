#-------------------------------------------Functions-------------------------------------------

def get_Embeddings(train_docs=[], test_docs=[]):
    import os
    import pickle

    fileName = "saveFiles/20NG_Embeddings.pkl"
    if os.path.exists(fileName):
        with open(fileName, 'rb') as temp:
            train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = pickle.load(temp)
    else:
        all_docs = train_docs
        all_docs.extend(test_docs)

        # Get Embeddings
        from Tools.Load_Embedings import Get_Embeddings
        embeddingGenerator = Get_Embeddings()
        doc_vectors, embeddings, maxSize, embedding_vocab = embeddingGenerator.googleVecs(all_docs)
        del embeddingGenerator
        train_doc_vectors = doc_vectors[:len(train_docs)]
        test_doc_vectors = doc_vectors[len(train_docs):]

        with open(fileName, 'wb') as temp:
            pickle.dump((train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab), temp)







#-------------------------------------------Prepare Data-------------------------------------------
## Get Dataset ##
from sklearn.datasets import fetch_20newsgroups

train_docs = fetch_20newsgroups(subset='train').data
train_labels = fetch_20newsgroups(subset='train').target
test_docs = fetch_20newsgroups(subset='test').data
test_labels = fetch_20newsgroups(subset='test').target
print("Total Doc Count : ", len(train_docs)+len(test_docs))

## Process Dataset ##
train_doc_vectors, test_doc_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(train_docs, test_docs)





# #-------------------------------------------Classification-------------------------------------------
# from sklearn.multiclass import OneVsRestClassifier
#
# """
# from sklearn.naive_bayes import GaussianNB
# classifier = OneVsRestClassifier(GaussianNB())
# #"""
# """
# from sklearn.svm import LinearSVC
# #classifier = OneVsRestClassifier(LinearSVC(tol=0.001, C=3.5, random_state=42))
# classifier = OneVsRestClassifier(LinearSVC(random_state=42))
# #"""
# """
# from sklearn.naive_bayes import MultinomialNB
# classifier = OneVsRestClassifier(MultinomialNB(alpha=.01))
# #"""
# """
# from sklearn.naive_bayes import MultinomialNB
# classifier = OneVsRestClassifier(MultinomialNB())
# #"""
# #"""
# from sklearn.neighbors import KNeighborsClassifier
# classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=15, n_jobs=-2))
# #"""
#
#
# classifier.fit(vectorised_train_documents, train_labels)
#
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
