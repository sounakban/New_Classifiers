#Take required user inputs
weight = "tf"
cor_type = input("Enter correlation coefficient in QUOTES [P for PMI, J for Jaccard] : ").upper()
if cor_type != "J" and cor_type != "P":
    raise ValueError ("Unrecognised option for correlation coefficient.")
#Lamda for Jelinek-Mercer Smoothing
lamda = 0.99
#Boost value of correlation-coefficients
coorelation_boost = 4
#Percentage of total term features to kepp
feature_percent = 100
#print "LAPLACE"




#Feature Extraction
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
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





start_time = time.time()
program_start = start_time

#----------------Get Corpus--------------------------

train_docs = fetch_20newsgroups(subset='train').data
test_docs = fetch_20newsgroups(subset='test').data

# Transform for multilabel compatibility
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([[v] for v in fetch_20newsgroups(subset='train').target.tolist()])
test_labels = mlb.transform([[v] for v in fetch_20newsgroups(subset='test').target.tolist()])

print len(train_docs), len(test_docs), train_labels.shape, test_labels.shape



#Create label complement matrix
train_labels_complement = numpy.zeros(shape=train_labels.shape);    train_labels_complement.fill(1)
train_labels_complement =  train_labels_complement - train_labels

num_labels = train_labels.shape[1]

#Get Class Prior Prob
#"""
priors = []
for i in range(num_labels):
    classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
    priors.append(len(classdoc_ids)/float(train_labels.shape[0]))
#"""

print "Building file list complete and it took : ", print_time(start_time)
start_time = time.time()



#----------------Feature Extraction--------------------------

#Process all documents
# Learn and transform documents [tf]
vectorizer_tf = CountVectorizer(stop_words=stop_words, tokenizer=tokenize)
vectorised_train_documents_tf = vectorizer_tf.fit_transform(train_docs)

term_freq = {}; term_prob = {}; all_terms = {}; totterms = 0.0

def add2dict(k, v, features):
    features[k]=  features.get(k, 0.0) + v


#Devide features by class
for i in range(num_labels):
    classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
    term_freq[i] = get_TF(vectorizer_tf, vectorised_train_documents_tf, classdoc_ids)
    map(lambda (k, v): add2dict(k, v, all_terms), term_freq[i].items())
    totterms += sum(term_freq[i].values())
all_terms_prob = {k: v/totterms for k, v in all_terms.items()}
#Devide features by class-complements
compl_term_freq = {}
for i in range(num_labels):
    classdoc_ids = numpy.nonzero(train_labels_complement[:, i])[0].tolist()
    compl_term_freq[i] = get_TF(vectorizer_tf, vectorised_train_documents_tf, classdoc_ids)
#Convert to Probability & Perform Jelinek-Mercer Smoothing
if weight == "tf" or cor_type == "P":
    for i in range(num_labels):
        term_prob[i] = freqToProbability(term_freq[i], compl_term_freq[i], all_terms_prob, lamda)
vocab_choice = term_prob


#Clear memory for unused variables
all_terms_list = all_terms.keys()
all_terms = {};     all_terms_prob = {};    vectorised_train_documents_tf = []

print "Generating term-weights complete and it took : ", print_time(start_time)
start_time = time.time()


#Find cooccurences for all classes and complements
if cor_type == "J":
    cooccurences_by_class = cooccurence_main.get_cooccurences(train_labels, train_docs, P_AandB=False)
elif cor_type == "P":
    cooccurences_by_class = cooccurence_main.get_cooccurences(train_labels, train_docs, P_AandB=True, term_freq=term_freq, term_prob=term_prob)

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

"""
#Perform feature selection on terms
from tools.cooccurence_utils import feature_selection
temp = {}
for i in range(num_labels):
    temp[0] = term_prob[i]
    temp[1] = term_prob[num_labels + i]
    feature_selection(temp, feature_list = all_terms_list, n_features = 0, percent = feature_percent)
    term_prob[i] = temp[0]
    term_prob[num_labels + i] = temp[1]
#"""



#----------------Classification--------------------------

classifier = CopulaClassifier(corcoeff, vocab_choice, priors)
predictions = classifier.predict_multiclass(test_docs)

print "The Classification is complete and it took", print_time(start_time)
#print "Avg time taken per doc: ", (print_time(start_time)/float(len(test_docs)))
start_time = time.time()


"""
print "Original:"
print test_labels
print "Predicted:"
print predictions
#"""




#-----------------Evaluation ----------------------
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

precision = precision_score(test_labels, predictions, average=None)
recall = recall_score(test_labels, predictions, average=None)
f1 = f1_score(test_labels, predictions, average=None)

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








"######################### Checking outputs #########################"

#print "Documents : ", getsizeof(documents)

#print numpy.nonzero(train_labels[4, :])[0].tolist()

"""
Rows: Docs ; Columns: Terms
print vectorised_test_documents_tfidf[[1, 3], :].shape
print vectorised_test_documents_tfidf.shape
print len(train_docs), " : ", vectorised_train_documents_tfidf.shape
print len(test_docs), " : ", vectorised_test_documents_tfidf.shape
"""

"""
print numpy.nonzero(term_freq[4].values())
print term_freq[4].keys()[9], " : ", term_freq[4].values()[9]
"""

"""
print "cooccurences_by_class : ", getsizeof(cooccurences_by_class)
print cooccurences_by_class[4].values()
"""

#print {k:v for k,v in cooccurences_by_class[2].items() if v<=0.0 or v>=1.0}

#print corcoeff[4].values()[:20]
