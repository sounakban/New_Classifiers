from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re       #for regular expression

import operator
import itertools
import numpy



cachedStopWords = stopwords.words("english")
def tokenize(text, word_length = 3):
    min_length = word_length
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter (lambda token: p.match(token) and
                               len(token) >= min_length, tokens))
    return filtered_tokens

"""
def freqToProbability(term_freq, complement_termfreq, all_term, lamda):
    if len(term_freq) <= 1:
        return {}
    vocab = term_freq.keys()
    tf_values = term_freq.values()
    #Divide by total freq
    tot = float(sum(tf_values))+1
    tf_values_array = numpy.array(tf_values)/tot
    prob_values = tf_values_array.tolist()
    term_prob = dict(itertools.izip(vocab, prob_values))
    #Repeat for complement set
    vocab = complement_termfreq.keys()
    tf_values = complement_termfreq.values()
    tot = float(sum(tf_values))+1
    tf_values_array = numpy.array(tf_values)/tot
    prob_values = tf_values_array.tolist()
    complement_term_prob = dict(itertools.izip(vocab, prob_values))
    #Perform Jelinek-Mercer Smoothing
    term_prob = {k: (term_prob.get(k, 0.0)*lamda + complement_term_prob.get(k, 0.0)*(1.0-lamda)) for k in all_term.keys()}
    if len(all_term) > len(term_freq) and len(all_term) > len(complement_termfreq):
        term_prob = {k: v for k, v in term_prob.items() if v != 0.0}
    return term_prob
#"""

#"""
def freqToProbability(term_freq, complement_termfreq, all_term, lamda):
    alpha = 0.01
    if len(term_freq) <= 1:
        return {}
    tf_values = term_freq.values()
    tot = float(sum(tf_values)) + (alpha*len(tf_values))

    term_prob = {k: (term_freq.get(k, 0.0) + alpha)/tot for k in all_term.keys()}
    if len(all_term) > len(term_freq) and len(all_term) > len(complement_termfreq):
        term_prob = {k: v for k, v in term_prob.items() if v != 0.0}
    return term_prob
#"""



def get_TF(vectorizer_tf, vectorised_document_tfs, doc_list):
    #vocab contains term-index pair
    vocab = vectorizer_tf.vocabulary_
    #sorted_vocab contains list of terms sorted on index
    sorted_vocab = [item[0] for item in sorted(vocab.items(), key=operator.itemgetter(1))]
    tf_values = numpy.array(vectorised_document_tfs[doc_list, :].sum(axis=0))[0].tolist()
    #vocab_tf_new is a dictionary that stores, freq of each term summed over all docs
    vocab_tf = dict(itertools.izip(sorted_vocab, tf_values))
    vocab_tf_new = {key: value for key, value in vocab_tf.items() if value != 0}
    return vocab_tf_new


def get_IDF(vectorizer_tfidf):
    idf = vectorizer_tfidf.idf_
    return dict(zip(vectorizer_tfidf.get_feature_names(), idf))


def get_TFIDF(vectorizer_tfidf, vectorised_train_documents_tfidf, doc_list):
    #vocab contains term-index pair
    vocab = vectorizer_tfidf.vocabulary_
    #sorted_vocab contains list of terms sorted based on index
    sorted_vocab = [item[0] for item in sorted(vocab.items(), key=operator.itemgetter(1))]
    #vocab_tfidf_new is a dictionary that stores tfidf sum over all docs for each term
    vocab_tfidf = dict(itertools.izip(sorted_vocab, numpy.array(vectorised_train_documents_tfidf[doc_list, :].sum(axis=0))[0].tolist()))
    vocab_tfidf_new = {key: value for key, value in vocab_tfidf.items() if value != 0}
    #print vocab_tfidf_new[sorted_vocab[-2]]
    return vocab_tfidf_new
