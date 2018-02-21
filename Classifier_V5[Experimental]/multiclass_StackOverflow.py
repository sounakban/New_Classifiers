#Take required user inputs
"""
weight = input("Enter weighing algorithm in QUOTES [tf or tfidf] : ").lower()
if weight != "tfidf" and weight != "tf":
    raise ValueError ("Unrecognised option for weighing algorithm.")
"""
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
# Percentage of Training Docs
train_cut = 0.8



#-----------------------Imports--------------------------

#Feature Extraction
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





#----------------Get Corpus--------------------------

start_time = time.time()
program_start = start_time



from tools.getStackOverflow import getStackOverflow
from random import sample

SOvrflow = getStackOverflow("/Volumes/Files/Work/Research/Information Retrieval/1) Data/StackOverflow-Dataset/")
data = SOvrflow.getData()
labels = SOvrflow.getTarget()


from sklearn.preprocessing import MultiLabelBinarizer

totrec = 0.0
totprec = 0.0
totF1 = 0.0
for i in range(10):
    train_indices = sample(range(len(data)), int(len(data)*train_cut) )
    test_indices = list( set(range(len(data))) - set(train_indices) )

    train_docs = [data[i] for i in train_indices]
    test_docs = [data[i] for i in test_indices]

    # Transform multilabel labels
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_labels)
    test_labels = mlb.transform(test_labels)


    #Create label-complement matrix
    train_labels_complement = numpy.zeros(shape=train_labels.shape);    train_labels_complement.fill(1)
    train_labels_complement =  train_labels_complement - train_labels

    #Get Class Prior Prob
    #"""
    priors = []
    for i in range(train_labels.shape[1]):
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
    #vectorised_test_documents_tf = vectorizer_tf.transform(test_docs)

    def add2dict(k, v, all_term):
        all_term[k]=  all_term.get(k, 0.0) + v

    #Devide features by class
    term_freq = {}; all_term = {}; tot_freq = 0.0
    for i in range(train_labels.shape[1]):
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        term_freq[i] = get_TF(vectorizer_tf, vectorised_train_documents_tf, classdoc_ids)
        map(lambda (k, v): add2dict(k, v, all_term), term_freq[i].items())
        tot_freq += sum(term_freq[i].values())
    all_term = {k: v/tot_freq for k, v in all_term.items()}
    #Devide features by class-complements
    compl_term_freq = {}
    for i in range(train_labels.shape[1]):
        classdoc_ids = numpy.nonzero(train_labels_complement[:, i])[0].tolist()
        compl_term_freq[i] = get_TF(vectorizer_tf, vectorised_train_documents_tf, classdoc_ids)

    #Convert to Probability & Perform Jelinek-Mercer Smoothing
    term_prob = {}
    if weight == "tf" or cor_type == "P":
        for i in range(train_labels.shape[1]):
            term_prob[i] = freqToProbability(term_freq[i], compl_term_freq[i], all_term, lamda)

    vocab_choice = term_prob

    #Clear memory for unused variables
    all_terms_list = all_term.keys()
    all_term = {};      vectorised_train_documents_tf = []

    print "Generating term-weights complete and it took : ", print_time(start_time)
    start_time = time.time()


    #Find cooccurences for all classes
    if cor_type == "J":
        cooccurences_by_class = cooccurence_main.get_cooccurences(train_labels, train_docs, P_AandB=False)
    elif cor_type == "P":
        cooccurences_by_class = cooccurence_main.get_cooccurences(train_labels, train_docs, P_AandB=True, term_freq=term_freq, term_prob=term_prob)

    print "Generating term-cooccurences complete and it took : ", print_time(start_time)
    start_time = time.time()


    #Find Correlation Coefficient Values
    if cor_type == "J":
        corcoeff = cooccurence_main.calc_corcoeff(cooccurences_by_class, term_freq, cor_type, boost = coorelation_boost/3)
    elif cor_type == "P":
        corcoeff = cooccurence_main.calc_corcoeff(cooccurences_by_class, term_prob, cor_type, boost = coorelation_boost)

    #Clear memory for unused variables
    cooccurences_by_class = []
    term_freq = []

    #"""
    #Perform feature selection on terms
    from tools.cooccurence_utils import feature_selection
    feature_selection(term_prob, feature_list = all_term.keys(), n_features = 0, percent = feature_percent)
    #"""

    print "Calculating correlation-coefficients complete and it took : ", print_time(start_time)
    start_time = time.time()




    #----------------Classification--------------------------

    classifier = CopulaClassifier(corcoeff, vocab_choice, priors)
    predictions = classifier.predict_multiclass(test_docs)

    print "The Classification is complete and it took", print_time(start_time)
    start_time = time.time()

    """
    print "Original:"
    print test_labels
    print "Predicted:"
    print predictions
    #"""




    #-----------------Evaluation ----------------------
    precision = precision_score(test_labels, predictions, average='micro')
    totprec += precision
    recall = recall_score(test_labels, predictions, average='micro')
    totrec += recall
    f1 = f1_score(test_labels, predictions, average='micro')
    totF1 += f1

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
    print("All-Class quality numbers")
    print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
            .format(precision, recall, f1))


print "10-fold Micro average:"
print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
.format(totprec/10, totrec/10, totF1/10))

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
