train_cut = 0.8

#-------------------------------------------Functions-------------------------------------------

def get_Embeddings(postweet=[], negtweet=[], selected_terms = set()):
    import os
    import pickle

    fileName = "saveFiles/Twit_Embeddings.pkl"
    if os.path.exists(fileName):
        with open(fileName, 'rb') as temp:
            postweet_vectors, negtweet_vectors, embeddings, maxSize, embedding_vocab = pickle.load(temp)
    else:
        all_docs = list(postweet)
        all_docs.extend(negtweet)

        # Get Embeddings
        from Tools.Load_Embedings import Get_Embeddings
        embeddingGenerator = Get_Embeddings()
        doc_vectors, embeddings, maxSize, embedding_vocab = embeddingGenerator.googleVecs(all_docs, selected_terms)
        del embeddingGenerator
		from keras.preprocessing.sequence import pad_sequences
		doc_vectors = pad_sequences(doc_vectors, maxlen=maxSize, padding='post', value=0.)
        postweet_vectors = doc_vectors[:len(postweet)]
        negtweet_vectors = doc_vectors[len(postweet):]

        with open(fileName, 'wb') as temp:
            pickle.dump((postweet_vectors, negtweet_vectors, embeddings, maxSize, embedding_vocab), temp)

	print("Embeddings Shape : ",embeddings.shape)
	return (postweet_vectors, negtweet_vectors, embeddings, maxSize, embedding_vocab)







#-------------------------------------------Prepare Data-------------------------------------------

from nltk.corpus import twitter_samples as tweet
from random import sample

postweet = tweet.strings('positive_tweets.json')
negtweet = tweet.strings('negative_tweets.json')

## Process Dataset ##
postweet_vectors, negtweet_vectors, embeddings, maxSize, embedding_vocab = get_Embeddings(postweet, negtweet)



totrec = 0.0
totprec = 0.0
totF1 = 0.0
K = 10
pos_set_size = len(postweet)/K
neg_set_size = len(negtweet)/K

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
