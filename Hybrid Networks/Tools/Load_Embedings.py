#Includes functions to import word vectors

import numpy as np

class Get_Embeddings:

	def __init__(self):
		import gensim
		self.google_vecs = gensim.models.KeyedVectors.load_word2vec_format('/Volumes/Files/Work/Research/Information Retrieval/General Tools/GoogleNews-vectors-negative300.bin', binary=True)
		# self.google_vecs = gensim.models.KeyedVectors.load_word2vec_format('./../../Resources/GoogleNews-vectors-negative300.bin', binary=True)
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
		self.max_wordCount = 0
		#Max number of sentences in a single document
		self.max_sentCount = 0


	def googleVecs(self, corpus, selected_terms):

		print("Starting vector and embeddings Generation")
		from nltk.tokenize import RegexpTokenizer, sent_tokenize
		tokenizer = RegexpTokenizer(r'\w+')

		not_in_vocab = 0
		for doc_num in range(len(corpus)):
			doc_temp = []
			for sentence in sent_tokenize(corpus[doc_num]):
				tokens = tokenizer.tokenize(sentence)
				tokens = [word for word in tokens if word in selected_terms]

				sent_temp = []
				for word in tokens:
					if word in self.embedding_vocab:
						sent_temp.append(self.embedding_vocab.index(word))

					elif word in self.google_vecs.vocab:
						sent_temp.append(self.num_of_words)
						self.embedding_vocab.append(word)
						self.embeddings.append(list(self.google_vecs[word]))
						self.num_of_words += 1

					else:
						not_in_vocab += 1
						sent_temp.append(self.num_of_words)
						self.embedding_vocab.append(word)
						self.embeddings.append(list(np.random.uniform(size=300)))
						self.num_of_words += 1

				if len(sent_temp) > self.max_wordCount:
					self.max_wordCount = len(sent_temp)
				doc_temp.append(sent_temp)

			if len(doc_temp) > self.max_sentCount:
				self.max_sentCount = len(doc_temp)
			self.doc_vectors.append(doc_temp)

		print("Load_Embedings :: GoogleVecs")
		print("Num of Docs : ", len(self.doc_vectors))
		print("Number of unique Words : ", self.num_of_words)
		print("Words not found in embeddings : ", not_in_vocab)
		print("Load_Embedings :: GoogleVecs")

		del self.google_vecs
		print("Completed vector and embeddings Generation")

		return (self.doc_vectors, np.array(self.embeddings), [self.max_sentCount, self.max_wordCount], self.embedding_vocab)
