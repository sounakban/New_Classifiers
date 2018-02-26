#-------------------------------------------Feature Extraction-------------------------------------------

from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1()
rcvdata = rcv1.data


vectorised_train_documents = rcvdata[ : 23149, : ]
#RCV1-V2 test-sets
#vectorised_test_documents = rcvdata[23149:222477, : ][:10000, :]
#vectorised_test_documents = rcvdata[23149:222477, : ][10000:20000, :]
#vectorised_test_documents = rcvdata[222477:421816, : ][:10000, :]
vectorised_test_documents = rcvdata[421816:621392, : ]
#vectorised_test_documents = rcvdata[621392:804414, : ][:10000, :]
#RCV1 Complete
#vectorised_test_documents = rcvdata[23149 : , : ]



train_labels = rcv1.target[ : 23149, : ]
#RCV1-V2 test-sets
#test_labels = rcv1.target[23149:222477, : ][:10000, :]
#test_labels = rcv1.target[23149:222477, : ][10000:20000, :]
#test_labels = rcv1.target[222477:421816, : ][:10000, :]
test_labels = rcv1.target[421816:621392, : ]
#test_labels = rcv1.target[621392:804414, : ][:10000, :]
#RCV1 Complete
#test_labels = rcv1.target[23149 : , : ]

print vectorised_train_documents.shape, vectorised_test_documents.shape, train_labels.shape, test_labels.shape


#-------------------------------------------Classification-------------------------------------------
from sklearn.multiclass import OneVsRestClassifier

#"""
from sklearn.naive_bayes import BernoulliNB
classifier = OneVsRestClassifier(BernoulliNB(alpha=0.01))
#"""
"""
from sklearn.svm import LinearSVC
classifier = OneVsRestClassifier(LinearSVC(random_state=42))
#"""
"""
from sklearn.naive_bayes import MultinomialNB
classifier = OneVsRestClassifier(MultinomialNB(alpha=0.01))
#"""
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=15, p=2, n_jobs=-2))
#"""


classifier.fit(vectorised_train_documents, train_labels)
print "training complete"
predictions = classifier.predict(vectorised_test_documents)


#-------------------------------------------Evaluation-------------------------------------------

from sklearn.metrics import f1_score, precision_score, recall_score

#MICRO
precision = precision_score(test_labels, predictions,
                            average='micro')
recall = recall_score(test_labels, predictions,
                      average='micro')
f1 = f1_score(test_labels, predictions, average='micro')

print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
        .format(precision, recall, f1))

#MACRO
precision = precision_score(test_labels, predictions,
                            average='macro')
recall = recall_score(test_labels, predictions,
                      average='macro')
f1 = f1_score(test_labels, predictions, average='macro')

print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
        .format(precision, recall, f1))

#INDIVIDUAL
precision = precision_score(test_labels, predictions,
                            average=None)
recall = recall_score(test_labels, predictions,
                      average=None)
f1 = f1_score(test_labels, predictions, average=None)

print("All-Class quality numbers")
print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
        .format(precision, recall, f1))

import numpy as np
exp_train = np.sum(train_labels.toarray(), axis=0)
exp_test = np.sum(test_labels.toarray(), axis=0)
print "Num of train docs per category:\n", exp_train
print "Num of test docs per category:\n", exp_test


#Export to Spreadsheet
import xlsxwriter

export = np.column_stack((exp_train, exp_test, f1, precision, recall))
workbook = xlsxwriter.Workbook('classscores.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "Train(Count)")
worksheet.write(0, 1, "Test(Count)")
worksheet.write(0, 2, "F1")
worksheet.write(0, 3, "Precision")
worksheet.write(0, 4, "Recall")
for (x,y), value in np.ndenumerate(export):
    worksheet.write(x+1, y, value)
workbook.close()
