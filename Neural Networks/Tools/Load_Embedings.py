#Includes functions to import word vectors

import numpy as np

class Get_Embeddings:

	def __init__(self):
		import gensim
		# self.google_vecs = gensim.models.KeyedVectors.load_word2vec_format('/Volumes/Files/Work/Research/Information Retrieval/General Tools/GoogleNews-vectors-negative300.bin', binary=True)
		self.google_vecs = gensim.models.KeyedVectors.load_word2vec_format('./../../Resources/GoogleNews-vectors-negative300.bin', binary=True)
		#List of words in vocabulary
		self.embedding_vocab = []
		#Mapping of doc words to embedding table
		self.doc_vectors = []
		#Embedding table
		self.embeddings = []
		#To keep the 0th element a random vector [to map the padded elements]
		self.embeddings.append(list(np.random.uniform(size=300)))
		self.num_of_words = 1	#0 is reserved for paddings
		#Max length of a single sentence
		self.maxSize = 0


	def googleVecs(self, corpus, selected_terms = set()):

		from nltk.tokenize import RegexpTokenizer
		tokenizer = RegexpTokenizer(r'\w+')
		# from nltk.stem import WordNetLemmatizer
		# wordnet_lemmatizer = WordNetLemmatizer()
		# from nltk.corpus import stopwords
		# stop_words = set(stopwords.words('english'))
		# if len(selected_terms) > 0:
		# 	selected_terms = set( [wordnet_lemmatizer.lemmatize(word.lower()) for word in selected_terms] )

		not_in_vocab = 0
		for doc_num in range(len(corpus)):
			tokens = tokenizer.tokenize(corpus[doc_num])
			# tokens = [wordnet_lemmatizer.lemmatize(word.lower()) for word in tokens if word in selected_terms and not word in stop_words]
			tokens = [word for word in tokens if word in selected_terms]

			doc_temp = []
			for word in tokens:
				if word in self.embedding_vocab:
					doc_temp.append(self.embedding_vocab.index(word))

				elif word in self.google_vecs.vocab:
					doc_temp.append(self.num_of_words)
					self.embedding_vocab.append(word)
					self.embeddings.append(list(self.google_vecs[word]))
					self.num_of_words += 1

				else:
					not_in_vocab += 1
					doc_temp.append(self.num_of_words)
					self.embedding_vocab.append(word)
					self.embeddings.append(list(np.random.uniform(size=300)))
					self.num_of_words += 1

			if len(doc_temp) > self.maxSize:
				self.maxSize = len(doc_temp)
			self.doc_vectors.append(doc_temp)

		print("Load_Embedings :: GoogleVecs_POS_triggerVecs")
		print("Num of Docs : ", len(self.doc_vectors))
		print("Number of unique Words : ", self.num_of_words)
		print("Words not found in embeddings : ", not_in_vocab)
		print("Load_Embedings :: GoogleVecs_POS_triggerVecs")

		del self.google_vecs

		return (self.doc_vectors, np.array(self.embeddings), self.maxSize, self.embedding_vocab)
