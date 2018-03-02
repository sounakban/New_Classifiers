import numpy as np

# Performs basic preprocessing : tokenization, stopword removal, lemmatizing
def preprocess(train_data, test_data):
	print("Starting data Preprocessing")
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

	print("Completed data Preprocessing")
	return(ret_train_data, ret_test_data)



def pad_sequences3D(doc_vectors, max_dim2, max_dim3, value=0):
	print("Starting 3D Padding")
	for doc in doc_vectors:
		while len(doc) < max_dim2:
			doc.append([value])
		for sent_vectors in doc:
			while len(sent_vectors) < max_dim3:
				sent_vectors.append(value)

	doc_vectors = np.array(doc_vectors)
	print("Completed 3D Padding")
	return doc_vectors
