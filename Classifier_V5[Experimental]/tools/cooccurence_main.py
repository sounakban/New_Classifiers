import multiprocessing
thread_count = int(multiprocessing.cpu_count()*0.75)
from functools import partial
import numpy
from cooccurence_extract import process_text, process_text_with_features
from cooccurence_utils import cal_PMI, get_P_AandB, cal_Jaccard, normalize_corcoeff, get_Simple_Prob, feature_selection


def add_pairs(cooc, (pair, value)):
    cooc[pair] = cooc.get(pair, 0) + value


def subtract_pairs(cooc, (pair, value)):
    cooc[pair] = cooc[pair] - value


def get_cooccurences(train_labels, train_docs, P_AandB, term_freq={}, term_prob={}):
    if P_AandB == True and (len(term_freq) != train_labels.shape[1] or len(term_prob) != train_labels.shape[1]):
        raise ValueError('@get_cooccurences: Either set P_AandB to False or pass a valid Term(Freq/Prob) dict, divided by class')

    from cooccurence_extract import process_text
    import numpy

    cooccurence_list = {}
    for i in range(train_labels.shape[1]):
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=100)
        classdocs = [train_docs[did] for did in classdoc_ids]
        file_coocs = pool.imap(process_text, classdocs)
        pool.close()
        pool.join()
        class_cooc = {}
        map(lambda fil_cooc: map(partial(add_pairs, class_cooc), fil_cooc.items()), file_coocs)
        cooccurence_list[i] = class_cooc
    print "All Cooccurences Generated."
    #feature_selection(cooccurence_list)
    #print "Feature Selection Complete"
    #convert from frozenset to tuple
    for i in range(len(cooccurence_list)):
        cooccurence_list[i] = {tuple(k):v for k,v in cooccurence_list[i].items()}
    #If user wants probability of occurence instead of raw-frequency
    if P_AandB == True:
        pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=10)

        #Option 1:
        #cooccurence_list_new = list(pool.map(get_Simple_Prob, cooccurence_list.values()))

        #Option 2:
        parameter = []
        for i in range(len(cooccurence_list)):
            parameter.append([cooccurence_list[i], term_freq[i], term_prob[i]])
        cooccurence_list_new = list(pool.map(get_P_AandB, parameter))

        pool.close()
        pool.join()
        cooccurence_list = cooccurence_list_new
        cooccurence_list = {i: v for i, v in enumerate(cooccurence_list)}
    #"""
    #Perform feature selection
    feature_selection(cooccurence_list)
    print "Feature Selection Complete"
    #"""
    cooccurence_list = list(cooccurence_list.values())
    return cooccurence_list



def get_cooccurences_BR(train_labels, train_docs, keep_terms = [], P_AandB = True, term_freq={}, term_prob={}):
    import sys
    if P_AandB == True and (len(term_freq) != 2*train_labels.shape[1] or len(term_prob) != 2*train_labels.shape[1]):
        raise ValueError('@get_cooccurences: Either set P_AandB to False or pass a valid Term(Freq/Prob) dict, divided by class')
    import itertools
    num_class = train_labels.shape[1]

    #Find Cooccurences accross all documents
    pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=100)
    if len(keep_terms) == 0:
        file_coocs = list(pool.map(process_text, train_docs))
    else:
        print "Keepping only co-occurence of terms from feature selected list"
        file_coocs = list(pool.map(partial(process_text_with_features, keep_terms), train_docs))
    pool.close()
    pool.join()
    total_cooc = {}
    map(lambda fil_cooc: map(partial(add_pairs, total_cooc), fil_cooc.items()), file_coocs)
    print "Training-Corpus Cooccurences Generated."
    #convert from frozenset to tuple
    #total_cooc = {tuple(k):v for k, v in total_cooc.items()}
    total_cooc_keys = [tuple(k) for k in total_cooc.keys()]

    #Find cooccurences for class
    cooccurence_list = {}
    for i in range(num_class):
        class_cooc = {}
        classdoc_ids = numpy.nonzero(train_labels[:, i])[0].tolist()
        class_files_cooc = list(map(lambda id: file_coocs[id], classdoc_ids))
        map(lambda fil_cooc: map(partial(add_pairs, class_cooc), fil_cooc.items()), class_files_cooc)

        #Generate complement class statictics by list subtraction

        #Process 1
        class_compl_cooc = dict(total_cooc)
        map(partial(subtract_pairs, class_compl_cooc), class_cooc.items())
        #class_compl_cooc = {k: v for k, v in class_compl_cooc.items() if v > 0}

        #Process 2
        """
        from collections import Counter
        class_cooc_counter = Counter(class_cooc)
        class_compl_cooc_counter = Counter(dict(total_cooc))
        class_compl_cooc_counter = class_compl_cooc_counter - class_cooc_counter
        class_compl_cooc = dict(class_compl_cooc_counter)
        class_cooc_counter = class_compl_cooc_counter = []
        #"""

        #Perform Processing
        temp = {}
        temp[0] = class_cooc
        temp[1] = class_compl_cooc
        feature_selection(temp, total_cooc.keys(), relative=True)
        temp[0] = {tuple(k): v for k, v in temp[0].items()}
        temp[1] = {tuple(k): v for k, v in temp[1].items()}
        if P_AandB == True:
            parameter = []
            parameter.append([temp[0], term_freq[i], term_prob[i]])
            parameter.append([temp[1], term_freq[num_class + i], term_prob[num_class + i]])
            pool = multiprocessing.Pool(processes=2)
            temp_new = list(pool.map(get_P_AandB, parameter))
            #temp_new = list(pool.map(get_Simple_Prob, temp_new))
            pool.close()
            pool.join()
        #feature_selection(temp, total_cooc_keys)
        #print "Original : \n", temp[0]
        #print "Compliment : \n", temp[1]
        cooccurence_list[i] = temp[0]
        cooccurence_list[num_class + i] = temp[1]
        if (i+1) % 5 == 0:
            print "Processed {} Classes.".format(i+1)
            sys.stdout.write("\033[F")
    print "All Cooccurences Generated."
    #Empty memory of unused variables
    file_coocs = []
    return cooccurence_list





def calc_corcoeff(cooccurence_list, vocab, cor_type, boost = 2):
    pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=10)
    parameter = []
    print len(cooccurence_list)
    for i in range(len(cooccurence_list)):
        parameter.append([cooccurence_list[i], vocab[i]])
    if cor_type == "P":
        cooccurence_list_new = list(pool.imap(cal_PMI, parameter))
    elif cor_type == "J":
        cooccurence_list_new = list(pool.imap(cal_Jaccard, parameter))
    pool.close()
    pool.join()
    cooccurence_list = cooccurence_list_new
    pool = multiprocessing.Pool(processes=thread_count, maxtasksperchild=10)
    new = pool.map(partial(normalize_corcoeff, boost), cooccurence_list)
    cooccurence_list = new
    pool.close()
    pool.join()
    """
    #For feature selection
    cooccurence_list = {i: v for i, v in enumerate(cooccurence_list)}
    print cooccurence_list.keys()
    feature_selection(cooccurence_list)
    cooccurence_list = list(cooccurence_list.values())
    print "Feature Selection Complete"
    #"""
    return cooccurence_list
