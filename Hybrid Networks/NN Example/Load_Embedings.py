#Includes functions to import word vectors

import numpy as np

class Embeddings:

	def __init__(self):
		import gensim
		self.google_vecs = gensim.models.KeyedVectors.load_word2vec_format('../../../Resources/GoogleNews-vectors-negative300.bin', binary=True)
		self.embedding_vocab = []
		self.doc_vectors = []
		self.embeddings = []
		#To keep the 0th element a random vector [to map the padded elements]
		self.embeddings.append(list(np.random.uniform(size=300)))
		self.POS_labels = []
		#To keep the 0th element an empty vector [to map the padded elemnts]
		#self.POS_labels.append(["."])
		self.num_of_words = 1	#0 is reserved for unknown words
		self.maxSize = 0


	def GoogleVecs_POS_triggerVecs(self, corpus, trigger_list):

		from Data_Standardization import oneHot_to_standard

		from nltk.tokenize import RegexpTokenizer
		tokenizer = RegexpTokenizer(r'\w+')
		from nltk.corpus import stopwords
		#stop_words = set(stopwords.words('english'))
		stop_words = set([])
		from nltk.stem import WordNetLemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()
		from nltk import pos_tag

		trig_vectors = []
		not_in_vocab = 0
		trig_not_in_vocab = 0
		tot_padding = 0
		for doc_num in range(len(corpus)):
			trig = tokenizer.tokenize(trigger_list[doc_num])
			trig = [wordnet_lemmatizer.lemmatize(t) for t in trig]
			if len(trig) == 1 and trig[0] in stop_words:
				continue
			tokens = tokenizer.tokenize(corpus[doc_num])
			tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens if not word in stop_words]
			tags = pos_tag(tokens)
			tags = [x[1] for x in tags]

			doc_temp = []
			trig_temp = []
			for word in tokens:
				if word in self.embedding_vocab:
					doc_temp.append(self.embedding_vocab.index(word))
					if word in trig:
						trig_temp.append([1,0])
					else:
						trig_temp.append([0,1])

				elif word in self.google_vecs.vocab:
					doc_temp.append(self.num_of_words)
					self.embedding_vocab.append(word)
					self.embeddings.append(list(self.google_vecs[word]))
					self.num_of_words += 1
					if word in trig:
						trig_temp.append([1,0])
					else:
						trig_temp.append([0,1])

				else:
					not_in_vocab += 1
					doc_temp.append(self.num_of_words)
					self.embedding_vocab.append(word)
					self.embeddings.append(list(np.random.uniform(size=300)))
					self.num_of_words += 1
					if word in trig:
						trig_not_in_vocab += 1
						trig_temp.append([1,0])
					else:
						trig_temp.append([0,1])

			#doc_temp, trig_temp, _ = oneHot_to_standard(doc_temp, trig_temp, tags)
			if len(doc_temp) > self.maxSize:
				self.maxSize = len(doc_temp)
			self.doc_vectors.append(doc_temp)
			trig_vectors.append(trig_temp)
			self.POS_labels.append(tags)

		print("Load_Embedings :: GoogleVecs_POS_triggerVecs")
		print("Num of Docs : ", len(corpus))
		print("Number of unique Words : ", self.num_of_words)
		print("Words not found in embeddings : ", not_in_vocab)
		print("Triggers not found in embeddings : ", trig_not_in_vocab)
		print("Total Padding : ", tot_padding)
		print("Load_Embedings :: GoogleVecs_POS_triggerVecs")

		del self.google_vecs

		return (self.doc_vectors, np.array(self.embeddings), trig_vectors, self.maxSize, self.POS_labels, self.embedding_vocab)

	# Perform BIO Encoding
	def GoogleVecs_POS_triggerVecs_merged(self, corpus, trigger_list):

		from nltk.tokenize import RegexpTokenizer
		tokenizer = RegexpTokenizer(r'\w+')
		from nltk.corpus import stopwords
		stop_words = set(stopwords.words('english'))
		#stop_words = set([])
		from nltk.stem import WordNetLemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()
		from nltk import pos_tag

		trig_vectors = []
		not_in_vocab = 0
		trig_not_in_vocab = 0
		tot_padding = 0
		for doc_num in range(len(corpus)):
			#trig = tokenizer.tokenize(trigger_list[doc_num])
			triggers = trigger_list[doc_num]
			triggers = [tokenizer.tokenize(trig) for trig in triggers]
			temp = []
			for trig in triggers:
				temp.append([wordnet_lemmatizer.lemmatize(x) for x in trig if not x in stop_words])
			triggers = temp
			if len(trig) == 1 and trig[0] in stop_words:
				continue
			tokens = tokenizer.tokenize(corpus[doc_num])
			tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens if not word in stop_words]
			tags = pos_tag(tokens)
			tags = [x[1] for x in tags]

			doc_temp = []
			trig_temp = []
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
					if word in trig:
						trig_not_in_vocab += 1

				if any(word in x for x in trig):
					for t in trig:
						if word in t:
							if t.index(word) == 0:
								trig_temp.append([1,0,0])
							else:
								trig_temp.append([0,1,0])
				else:
					trig_temp.append([0,0,1])

			#doc_temp, trig_temp, _ = oneHot_to_standard(doc_temp, trig_temp, tags)
			if len(doc_temp) > self.maxSize:
				self.maxSize = len(doc_temp)
			self.doc_vectors.append(doc_temp)
			trig_vectors.append(trig_temp)
			self.POS_labels.append(tags)

		print("Load_Embedings :: GoogleVecs_POS_triggerVecs")
		print("Num of Docs : ", len(corpus))
		print("Number of unique Words : ", self.num_of_words)
		print("Words not found in embeddings : ", not_in_vocab)
		print("Triggers not found in embeddings : ", trig_not_in_vocab)
		print("Total Padding : ", tot_padding)
		print("Load_Embedings :: GoogleVecs_POS_triggerVecs")

		del self.google_vecs

		return (self.doc_vectors, np.array(self.embeddings), trig_vectors, self.maxSize, self.POS_labels, self.embedding_vocab)
