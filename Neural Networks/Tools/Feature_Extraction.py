# TAKES ALL DOCUMENTS AND BINARIZED LABELS AS INPUT
# AND RETURNS SET OF TERMS THAT HAVE BEEN SELECTED

#---------------------------------------Preprocessing Functions---------------------------------------

def vectorize_tfidf(docs):
	from sklearn.feature_extraction.text import TfidfVectorizer
	vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.2, stop_words=None)

	X = vectorizer.fit_transform(docs)
	vocabulary2index_dict = vectorizer.vocabulary_
	index2vocabulary_dict = {v : k for k, v in vocabulary2index_dict.items()}
	ignored_words = vectorizer.stop_words_

	return (X, index2vocabulary_dict, ignored_words)






#---------------------------------------Main Functions---------------------------------------

def chisqure(train_docs, train_labels, feature_count = 500):

	train_docs_vectorized, index2vocabulary_dict, ignored_words = vectorize_tfidf(train_docs)

	if feature_count == 0:
		# feature_count = train_docs_vectorized.shape[1]
		return index2vocabulary_dict.values()

	from sklearn.feature_selection import SelectKBest, chi2
	feature_selector = SelectKBest(chi2, k=feature_count)
	docs_new = feature_selector.fit_transform(train_docs_vectorized, train_labels)
	feature_mask = kept_features = feature_selector.get_support(indices=False)
	tot_feature_count = feature_mask.shape[0]
	# removed_features = set( [index2vocabulary_dict[i] for i in range(tot_feature_count) if feature_mask[i]==False] )
	# all_ignored = ignored_words.union(removed_features)
	selected_features = set( [index2vocabulary_dict[i] for i in range(tot_feature_count) if feature_mask[i]==True] )

	return selected_features
