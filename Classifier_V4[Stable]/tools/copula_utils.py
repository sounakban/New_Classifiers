from cooccurence_extract import process_text, process_text_with_features
from math import log, exp
import numpy
import functools


def classify(corcoeff, vocab, priors, test_docs, num, que, all_terms = []):
    if len(priors)==0:
        log_priors = [0]*len(corcoeff)
    else:
        log_priors = numpy.ma.log(numpy.array(priors)).tolist()
    all_scores = [num]
    for doc in test_docs:
        scorelist = []
        if len(all_terms) > 0:
            doc_repr = process_text_with_features(all_terms, doc)
        else:
            doc_repr = process_text(doc)
        cooccurences = {tuple(k):v for k, v in doc_repr.items()}
        for i in range(len(corcoeff)):
            if len(corcoeff[i]) == 0:
                score = 0
            else:
                #sub_scores = []
                curr_coeffs = corcoeff[i]
                curr_vocab = vocab[i]
                foo = functools.partial(get_scores, curr_coeffs, curr_vocab)
                sub_scores = map(foo, cooccurences.keys())
                sub_scores = list(filter(None, sub_scores))
                sub_scores = numpy.log(numpy.array(sub_scores))
                #priors = (numpy.array(priors)/min(priors)).tolist()
                score = numpy.sum(sub_scores)+int(log_priors[i] or 0)
                #print score, len(sub_scores)
            scorelist.append(score)
        #print sorted(scorelist, reverse=False)
        #print scorelist
        all_scores.append(scorelist)
        doc = ""
    que.put(all_scores)




#----------------------Common Functions----------------------

def get_scores(curr_coeffs, curr_vocab, pair):
    if pair[0] not in curr_vocab or pair[1] not in curr_vocab:
        return 0.0;
    theta = curr_coeffs.get(pair, 1)
    return bivariate_gumbel(curr_vocab[pair[0]], curr_vocab[pair[1]], theta)

def bivariate_gumbel(p1, p2, theta):
    res = phi_inv_gumbel(phi_gumbel(p1, theta) + phi_gumbel(p2, theta), theta)
    if res <= 0.0:
        print "@bivariate_gumbel: ", p1, p2, theta, res
    return res

def phi_gumbel(termWeight, theta):
    return (-log(termWeight))**theta

def phi_inv_gumbel(termWeight, theta):
    return exp( -(termWeight ** (1/theta)) )


def pred_maxScore(scores):
    pred = [0]*len(scores)
    pred[scores.index(max(scores))] = 1
    return numpy.array(pred)


def pred_ScoreBR(scores):
    num_classes = len(scores)/2
    classes = [i for i in range(num_classes) if scores[i] > scores[num_classes + i] and scores[i] != 0 and scores[num_classes + i] != 0]
    pred = [0]*num_classes
    for i in classes:
        pred[i] = 1
    return pred
