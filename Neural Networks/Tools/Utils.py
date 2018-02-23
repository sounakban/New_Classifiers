# Performs basic preprocessing : tokenization, stopword removal, lemmatizing
def preprocess(train_data, test_data):
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    ret_train_data = []
    for sample in train_data:
		tokens = tokenizer.tokenize(sample)
        tokens = [wordnet_lemmatizer.lemmatize(word.lower()) for word in tokens if not word in stop_words]
        ret_train_data.append(" ".join(tokens))

    ret_test_data = []
    for sample in test_data:
		tokens = tokenizer.tokenize(sample)
        tokens = [wordnet_lemmatizer.lemmatize(word.lower()) for word in tokens if not word in stop_words]
        ret_test_data.append(" ".join(tokens))

    return(ret_train_data, ret_test_data)
