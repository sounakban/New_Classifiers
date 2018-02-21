import numpy
import itertools
import math
from functools import partial





################################## For finding Cooccurences ##################################

def get_Simple_Prob(cooccur):
    tot = float(numpy.sum(numpy.array(cooccur.values())))
    values = (numpy.array(cooccur.values())/tot).tolist()
    res = dict(itertools.izip(cooccur.keys(), values))
    return res


def get_P_AandB(parameter):
    cooccur = parameter[0]
    vocab_tf = parameter[1]
    vocab_tprob = parameter[2]
    #Original Formula [Resource Heavy]
    map(partial(P_AandB_formula, cooccur, vocab_tf, vocab_tprob), cooccur.keys())

    return cooccur

def P_AandB_formula(cooccur, vocab_tf, vocab_tprob, pair):
    if cooccur[pair] == 0:
        print "@P_AandB_formula, 0 value: ", cooccur[pair]
        del cooccur[pair]
    elif pair[0] in vocab_tf and pair[1] in vocab_tf:
        P_BgivenA = cooccur[pair]/float(vocab_tf[pair[0]])
        P_AandB = vocab_tprob[pair[0]] * P_BgivenA
        if(P_AandB == 0.0):
            print pair, ": {}/{}= {} | {}".format(cooccur[pair], vocab_tf[pair[0]], P_BgivenA, vocab_tprob[pair[0]])
        elif(P_AandB <= 1):
            cooccur[pair] = P_AandB
        else:
            print "P_AandB > 1, @ get_P_AandB"
            del cooccur[pair]
    else:
        del cooccur[pair]

def feature_selection(all_features, feature_list = [], n_features = 200, n_percent = 0, percent = 0, relative=False):
    import operator

    #If user does not provide the list of all features
    if len(feature_list) == 0:
        for coouc in all_features.values():
            feature_list.extend(coouc.keys())
        #Remove duplicates
    feature_list = list(set(feature_list))

    #Create matrix for chi-squared test
    tot = [1]*len(all_features)
    if relative == True:
        for i in range(len(all_features)):
            tot[i] = numpy.sum(all_features[i].values())+1

    matrix = numpy.zeros(shape=(len(all_features),len(feature_list)))

    for i in range(len(all_features)):
        matrix[i] = map(lambda pair: all_features[i].get(pair, 0)/tot[i], feature_list)
        #Set n_features to the min length of features if feature len for some class is less
        if len(all_features[i]) < n_features and len(all_features[i]) > 1:
            n_features = len(all_features[i])

    #Select features using chi-squared test
    from sklearn.feature_selection import SelectKBest, chi2

    if percent != 0:
        selector = SelectKBest(chi2, k = int(len(feature_list)*percent)/100 )
        matrix = selector.fit_transform(matrix, range(len(all_features)))
        #Clean unused memory
        matrix = []

        kept_features = selector.get_support(indices=True)
        feature_list = [feature_list[i] for i in kept_features]
        for i in range(len(all_features)):
            all_features[i] = {pair: all_features[i].get(pair, 0) for pair in feature_list}
            all_features[i] = {k: v for k, v in all_features[i].items() if v!=0}
        """
        #Select top N candidates
        if n_features != 0:
            for i in range(len(all_features)):
                if len(all_features[i]) > n_features:
                    all_features[i] = dict(sorted(all_features[i].iteritems(), key=operator.itemgetter(1), reverse=True)[:n_features])
        #"""

    elif n_features != 0:
        chi2, pval = chi2(matrix, range(len(all_features)))
        pval_dict = dict(itertools.izip(feature_list, pval))
        """
        for i in range(len(all_features)):
            curr_features = sorted(all_features[i].keys(), key =lambda x:  pval_dict[x])[:n_features]
            all_features[i] = {pair: all_features[i][pair] for pair in curr_features}
        """
        feature_list_sorted = sorted(pval_dict, key=pval_dict.get)
        for i in range(len(all_features)):
            curr_features = {}
            for k in feature_list_sorted:
                val = all_features[i].get(k, 0)
                if val > 0:
                    curr_features[k] = val
                    if len(curr_features) == n_features:
                        break
            all_features[i] = curr_features
        #print len( set(all_features[0].keys()) & set(all_features[1].keys()) )

    elif n_percent != 0:
        chi2, pval = chi2(matrix, range(len(all_features)))
        pval_dict = dict(itertools.izip(feature_list, pval))
        feature_lim = int( (min([len(all_features[i]) for i in range(len(all_features))]) * n_percent)/100 )
        print feature_lim, [len(all_features[i]) for i in range(len(all_features))]

        feature_list_sorted = sorted(pval_dict, key=pval_dict.get)
        for i in range(len(all_features)):
            curr_features = {}
            for k in feature_list_sorted:
                val = all_features[i].get(k, 0)
                if val > 0:
                    curr_features[k] = val
                    if len(curr_features) == feature_lim:
                        break
            all_features[i] = curr_features

    else:
        raise ValueError('@feature_selection : Enter non-zero value for n_features or percent')





################################## For Correlation-Coefficients ##################################

def cal_PMI(input_lists):
    curr_cooccur = input_lists[0]
    curr_vocab = input_lists[1]
    #Calculate PMI Coefficients
    map(partial(PMI_furmula, curr_cooccur, curr_vocab), curr_cooccur.items())
    return curr_cooccur

def PMI_furmula(curr_cooccur, curr_vocab, (pair, value)):
    if pair[0] in curr_vocab and pair[1] in curr_vocab:
        ValueCheck = value/float(curr_vocab[pair[0]]*curr_vocab[pair[1]])
        if ValueCheck <= 0.0:
            print pair, ValueCheck, ":", value, curr_vocab[pair[0]], curr_vocab[pair[1]]
        curr_cooccur[pair] = math.log(ValueCheck, 2)
        if curr_cooccur[pair] < 0.0:
            print "@pmi_formula: ", curr_vocab[pair[0]], curr_vocab[pair[1]], value
    else:
        del curr_cooccur[pair]



def cal_Jaccard(input_lists):
    curr_cooccur = input_lists[0]
    curr_vocab = input_lists[1]
    #Calculate Jaccard Coefficients
    map(partial(Jaccard_formula, curr_cooccur, curr_vocab), curr_cooccur.keys())
    return curr_cooccur

def Jaccard_formula(curr_cooccur, curr_vocab, pair):
    if pair[0] in curr_vocab and pair[1] in curr_vocab:
        divByZerocheck = curr_vocab[pair[0]]+curr_vocab[pair[1]]-curr_cooccur[pair]
        if(divByZerocheck > 0):
            curr_cooccur[pair] = curr_cooccur[pair]/float(divByZerocheck)
        else:
            print "@Jaccard_formula Cooccurence too high {}, w1: {}, w2: {}, freq: {}".format(pair, curr_vocab[pair[0]], curr_vocab[pair[1]], curr_cooccur[pair])
            del curr_cooccur[pair]
    else:
        del curr_cooccur[pair]

def normalize_corcoeff(boost, cooccurence_list):
    curr_list = cooccurence_list
    value_list = curr_list.values()
    if (len(value_list) == 0):
        print "Length: ", len(value_list)
    mean = float(sum(value_list))/(len(value_list) + 1)
    if mean == 0:
        print "Mean: ", mean
    for pair in curr_list.keys():
        if curr_list[pair] <= mean:
            del curr_list[pair]
        else:
            curr_list[pair] = (curr_list[pair]/mean)**boost
            if curr_list[pair] > 100.0 or curr_list[pair] < 1.0:
                print "@normalize: ", pair, curr_list[pair]
    return curr_list
