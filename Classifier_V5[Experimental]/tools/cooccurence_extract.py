from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import re
from functools import partial
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import itertools
#import csv




def add_to_dict(all_pairs, pair):
    if(len(pair)==2):
        pair_set = frozenset(pair)
        all_pairs[pair_set] = 1 + all_pairs.get(pair_set, 0)



#Return clean sentence
stop = stopwords.words('english') + list(string.punctuation)
def clean_text(sent, word_length = 3):
    words = [word.lower() for word in word_tokenize(sent) if word.lower() not in stop]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= word_length, tokens))
    return filtered_tokens

def process_text(text):
    all_pairs = {}
    #writeFile1 = csv.writer(open('Cooccur.csv', 'w'))

    sent_tokenize_list = sent_tokenize(text)
    for sent in sent_tokenize_list:
        words = clean_text(sent)

        #Eliminate duplicate words to get entence level cooc
        words = list(set(words))
        word_pairs = list(itertools.combinations(words, 2))
        map(partial(add_to_dict, all_pairs), word_pairs)
    #print "writing results to file : Cooccur.csv"
    #for key, val in all_pairs.items():
    #    writeFile1.writerow([key, val])
    return all_pairs



#Return clean sentence
def clean_text_with_features(keep_terms, sent, word_length = 3):
    words = [word.lower() for word in word_tokenize(sent) if word.lower() not in stop]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    filtered_tokens = ([w for w in tokens if w in keep_terms])
    #p = re.compile('[a-zA-Z]+');
    #filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= word_length, filtered_tokens))
    return filtered_tokens


def process_text_with_features(keep_terms, text):
    all_pairs = {}
    #writeFile1 = csv.writer(open('Cooccur.csv', 'w'))

    sent_tokenize_list = sent_tokenize(text)
    for sent in sent_tokenize_list:
        words = clean_text_with_features(keep_terms, sent)

        #Eliminate duplicate words to get entence level cooc
        words = list(set(words))
        word_pairs = list(itertools.combinations(words, 2))
        map(partial(add_to_dict, all_pairs), word_pairs)
    #print "writing results to file : Cooccur.csv"
    #for key, val in all_pairs.items():
    #    writeFile1.writerow([key, val])
    return all_pairs
