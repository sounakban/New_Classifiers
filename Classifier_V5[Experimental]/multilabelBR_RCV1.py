###### Copula Classification with Jalinek-Mercer Smoothing #######

#Take required user inputs
weight = "tf"
cor_type = input("Enter correlation coefficient in QUOTES [P for PMI, J for Jaccard] : ").upper()
if cor_type != "J" and cor_type != "P":
    raise ValueError ("Unrecognised option for correlation coefficient.")
#Lamda for Jelinek-Mercer Smoothing
lamda = 0.85
#Boost value of correlation-coefficients
coorelation_boost = 8
#Percentage of total term features to kepp
feature_percent = 4



#-------------------------- Begin Program --------------------------


#Corpus Data
from sklearn.datasets import fetch_rcv1
rcv1_info = fetch_rcv1()
sklearn_labelMatrix = rcv1_info.target.toarray()
sklearn_docIDs = rcv1_info.sample_id
rcv1_info = []
RCV1V2Path = "/Volumes/Files/Work/Research/Information Retrieval/1) Data/Reuters/RCV/RCV/RCV1-V2/Raw Data/"

from tools.getRCV1V2 import getRCV1V2
rcv1v2_data = getRCV1V2(RCV1V2Path, testset=1)

from tools.getRCV1 import getRCV1
RCV1Path = "/Volumes/Files/Work/Research/Information Retrieval/1) Data/Reuters/RCV/RCV/rcv1/Data/"
rcv1_data = getRCV1(RCV1Path, RCV1V2Path, testset=1)

#Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
import tools.cooccurence_main as cooccurence_main
from tools.text_processing import tokenize, get_TF, get_TFIDF, freqToProbability
import numpy

#Classification
from tools.CopulaClassifier import CopulaClassifier

#Evaluation
from sklearn.metrics import f1_score, precision_score, recall_score

#Common
from sys import getsizeof
import time
def print_time(start_time):
    tm = time.time() - start_time
    if tm > 100:
        return "{} minuites".format(tm/60.0)
    else:
        return "{} seconds".format(tm)




#-------------------------- Get Corpus --------------------------

start_time = time.time()
program_start = start_time


train_docs_index = range(rcv1_data.getTrainDocCount())
print "Num of train Docs: ", len(train_docs_index)
test_docs_index = range(rcv1_data.getTrainDocCount(), rcv1_data.getTotalDocCount())
print "Num of test Docs: ", len(test_docs_index)
#test_docs_index = [test_docs_index[2]]

documents = rcv1_data.getData()
train_docs = [documents[doc_ind] for doc_ind in train_docs_index]
train_docsv2 = [rcv1v2_data.getData()[doc_ind] for doc_ind in train_docs_index]
test_docs = [documents[doc_ind] for doc_ind in test_docs_index]
documents = []

#Get Doc-Label Matrices
test_docIDs = [rcv1_data.getDocIDs()[i] for i in test_docs_index]
sklearn_testIndices = []
sklearn_testIndices.append(numpy.where(sklearn_docIDs == test_docIDs[0])[0][0])
i = sklearn_testIndices[0]+1; j = 1
while(j<len(test_docIDs)):
    if sklearn_docIDs[i] == test_docIDs[j]:
        sklearn_testIndices.append(i)
        i+=1; j+=1
    else:
        print "Disparity at test doc num: ", j
        break
train_labels = numpy.array([sklearn_labelMatrix[i] for i in train_docs_index])
test_labels = numpy.array([sklearn_labelMatrix[i] for i in sklearn_testIndices])

#Create label complement matrix
train_labels_complement = numpy.zeros(shape=train_labels.shape);    train_labels_complement.fill(1)
train_labels_complement =  train_labels_complement - train_labels

num_labels = train_labels.shape[1]

#Get Class Prior Prob
priors = []
for i in range(num_labels):
    classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
    priors.append(len(classdoc_ids)/float(train_labels.shape[0]))
for i in range(num_labels):
    classdoc_ids = numpy.nonzero(train_labels_complement[:, i])[0].tolist()
    priors.append(len(classdoc_ids)/float(train_labels.shape[0]))


#Unused variables
sklearn_labelMatrix = []; sklearn_testIndices = []; sklearn_docIDs = []; rcv1_data = []

print "Building file list complete and it took : ", print_time(start_time)
start_time = time.time()

#Clear memory
rcv1_data = []
rcv1v2_data = []

#-------------------------- Feature Extraction --------------------------

#Process all documents
# Learn and transform documents [tf]
vectorizer_tf = CountVectorizer(stop_words=stop_words, tokenizer=tokenize)
vectorised_train_documents_tf = vectorizer_tf.fit_transform(train_docs)
train_docsv2 = []
#vectorised_test_documents_tf = vectorizer_tf.transform(test_docs)

term_freq = {}; term_prob = {}; all_terms = {}; totterms = 0.0

def add2dict(k, v, features):
    features[k]=  features.get(k, 0.0) + v

#Devide features by class
for i in range(num_labels):
    classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
    if len(classdoc_ids) == 0:
        term_freq[i] = {term_freq[i-1].keys()[0]: 0}
        continue
    term_freq[i] = get_TF(vectorizer_tf, vectorised_train_documents_tf, classdoc_ids)
    map(lambda (k, v): add2dict(k, v, all_terms), term_freq[i].items())
    totterms += sum(term_freq[i].values())
#all_terms_prob = {k: v/totterms for k, v in all_terms.items()}
all_terms_prob = all_terms
#Devide features by class-complements
for i in range(num_labels):
    classdoc_ids = numpy.nonzero(train_labels_complement[:, i])[0].tolist()
    term_freq[num_labels + i] = get_TF(vectorizer_tf, vectorised_train_documents_tf, classdoc_ids)
#Convert to Probability & Perform Jelinek-Mercer Smoothing
if weight == "tf" or cor_type == "P":
    for i in range(num_labels):
        term_prob[i] = freqToProbability(term_freq[i], term_freq[num_labels + i], all_terms_prob, lamda)
        term_prob[num_labels + i] = freqToProbability(term_freq[num_labels + i], term_freq[i], all_terms_prob, lamda)
vocab_choice = term_prob

#Clear memory for unused variables
all_terms_list = all_terms.keys()
all_terms = {}; all_terms_prob = {}; vectorised_train_documents_tf = []

#Perform feature selection on terms
from tools.cooccurence_utils import feature_selection
temp = {}
for i in range(num_labels):
    temp[0] = term_prob[i]
    temp[1] = term_prob[num_labels + i]
    #feature_selection(temp, feature_list = all_terms_list, n_features = 0, percent = feature_percent)
    feature_selection(temp, feature_list = all_terms_list, n_features = 5000)
    term_prob[i] = temp[0]
    term_prob[num_labels + i] = temp[1]

all_terms_list = []
for i in range(2*num_labels):
    term_freq[i] = {k: v for k, v in term_freq[i].items() if k in term_prob[i]}
    all_terms_list.extend(term_prob[i].keys())
all_terms_list = set(all_terms_list)

print len(all_terms_list)

print "Generating term-weights complete and it took : ", print_time(start_time)
start_time = time.time()


#Find cooccurences for all classes and complements
if cor_type == "J":
    cooccurences_by_class = cooccurence_main.get_cooccurences_BR(train_labels, train_docs, keep_terms = all_terms_list, P_AandB=False)
elif cor_type == "P":
    cooccurences_by_class = cooccurence_main.get_cooccurences_BR(train_labels, train_docs, keep_terms = all_terms_list, P_AandB=True, term_freq=term_freq, term_prob=term_prob)

print "Generating term-cooccurences complete and it took : ", print_time(start_time)
start_time = time.time()


#Find Correlation Coefficient Values
if cor_type == "J":
    corcoeff = cooccurence_main.calc_corcoeff(cooccurences_by_class, term_freq, cor_type, boost = coorelation_boost)
elif cor_type == "P":
    corcoeff = cooccurence_main.calc_corcoeff(cooccurences_by_class, term_prob, cor_type, boost = coorelation_boost)

#Clear memory for unused variables
cooccurences_by_class = []
term_freq = []

print "Calculating correlation-coefficients complete and it took : ", print_time(start_time)
start_time = time.time()




#-------------------------- Classification --------------------------

classifier = CopulaClassifier(corcoeff, vocab_choice, priors)
predictions = classifier.predict_multilabelBR(test_docs, all_terms = all_terms_list)

print "The Classification is complete and it took", print_time(start_time)
#print "Avg time taken per doc: ", (print_time(start_time)/float(len(test_docs)))
start_time = time.time()




#-------------------------- Evaluation ----------------------
precision = precision_score(test_labels, predictions, average='micro')
recall = recall_score(test_labels, predictions, average='micro')
f1 = f1_score(test_labels, predictions, average='micro')

print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

precision = precision_score(test_labels, predictions, average='macro')
recall = recall_score(test_labels, predictions, average='macro')
f1 = f1_score(test_labels, predictions, average='macro')

print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))


#print [len(numpy.nonzero(train_labels[:, i])[0].tolist()) for i in range(num_labels)]
#print [len(numpy.nonzero(test_labels[:, i])[0].tolist()) for i in range(num_labels)]
precision = precision_score(test_labels, predictions, average=None)
recall = recall_score(test_labels, predictions, average=None)
f1 = f1_score(test_labels, predictions, average=None)
print f1

print "Evaluation complete and it took : ", print_time(start_time)


import numpy as np
exp_train = np.sum(train_labels, axis=0)
exp_test = np.sum(test_labels, axis=0)
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




print "Total time taken : ", (time.time() - program_start)/60.0, "minuites"
