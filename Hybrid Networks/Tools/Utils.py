# Performs basic preprocessing : tokenization, stopword removal, lemmatizing
def preprocess(train_data, test_data):
	from nltk.tokenize import RegexpTokenizer, sent_tokenize
	tokenizer = RegexpTokenizer(r'\w+')
	from nltk.stem import WordNetLemmatizer
	wordnet_lemmatizer = WordNetLemmatizer()
	from nltk.corpus import stopwords
	stop_words = set(stopwords.words('english'))

	ret_train_data = []
	for sample in train_data:
		doc_temp = ""
		for sentence in sent_tokenize(sample):
			tokens = tokenizer.tokenize(sentence)
			tokens = [wordnet_lemmatizer.lemmatize(word.lower()) for word in tokens if not word in stop_words]
			doc_temp = doc_temp + " " + " ".join(tokens) + "."
		ret_train_data.append(doc_temp.strip())

	ret_test_data = []
	for sample in test_data:
		doc_temp = ""
		for sentence in sent_tokenize(sample):
			tokens = tokenizer.tokenize(sentence)
			tokens = [wordnet_lemmatizer.lemmatize(word.lower()) for word in tokens if not word in stop_words]
			doc_temp = doc_temp + " " + " ".join(tokens) + "."
		ret_test_data.append(doc_temp.strip())

	return(ret_train_data, ret_test_data)
