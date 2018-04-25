#####################################
## Contains Classification Modules ##
#####################################

#-----------------------------------Common Functions & Imports-------------------------------------

import numpy as np
# np.random.seed(123456)
from tensorflow import set_random_seed
# set_random_seed(2017)
from keras.backend import int_shape

def test_model(model, X_test, Y_test, Y_train):
	from keras.utils import to_categorical
	#Get Class of max predicted value
	predictions = model.predict(X_test).argmax(axis=-1)
	#Convert to binary category matrix with int type
	predictions = np.array(to_categorical(predictions, num_classes=Y_test.shape[1]), dtype=np.int16)

	from sklearn.metrics import f1_score, precision_score, recall_score
	test_labels = Y_test

	#MICRO
	print("Shape of \n1.test_labels : ", test_labels.shape, "\n2.predictions : ", predictions.shape)
	precision = precision_score(test_labels, predictions, average='micro')
	recall = recall_score(test_labels, predictions, average='micro')
	f1 = f1_score(test_labels, predictions, average='micro')

	print("Micro-average quality numbers")
	print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
			.format(precision, recall, f1))

	# totprec += precision
	# totrec += recall
	# totF1 += f1

	#MACRO
	precision = precision_score(test_labels, predictions, average='macro')
	recall = recall_score(test_labels, predictions, average='macro')
	f1 = f1_score(test_labels, predictions, average='macro')

	print("Macro-average quality numbers")
	print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
			.format(precision, recall, f1))

	#INDIVIDUAL
	precision = precision_score(test_labels, predictions, average=None)
	recall = recall_score(test_labels, predictions, average=None)
	f1 = f1_score(test_labels, predictions, average=None)

	print("All-Class quality numbers")
	print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
			.format(precision, recall, f1))

	# print "K-fold Micro average:"
	# print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
	#         .format(totprec/10, totrec/10, totF1/10))


	import numpy as np
	exp_train = np.sum(Y_train, axis=0)
	exp_test = np.sum(Y_test, axis=0)
	print "Num of train docs per category:\n", exp_train
	print "Num of test docs per category:\n", exp_test


	#Export to Spreadsheet
	import xlsxwriter

	export = np.column_stack((exp_train, exp_test, f1, precision, recall))
	workbook = xlsxwriter.Workbook('classscores.xlsx')
	worksheet = workbook.add_worksheet()
	worksheet.write(0, 0, "Train(Count)")
	worksheet.write(0, 1, "Test(Count)")
	worksheet.write(0, 2, "F1")
	worksheet.write(0, 3, "Precision")
	worksheet.write(0, 4, "Recall")
	for (x,y), value in np.ndenumerate(export):
	    worksheet.write(x+1, y, value)
	workbook.close()


def test_model_multilabel(model, X_test, Y_test, Y_train):
	predictions = model.predict(X_test)
	# Convert to binary based on probability
	predictions[predictions>=0.5] = 1
	predictions[predictions<0.5] = 0

	from sklearn.metrics import f1_score, precision_score, recall_score
	test_labels = Y_test

	#MICRO
	print("Shape of \n1.test_labels : ", test_labels.shape, "\n2.predictions : ", predictions.shape)
	precision = precision_score(test_labels, predictions, average='micro')
	recall = recall_score(test_labels, predictions, average='micro')
	f1 = f1_score(test_labels, predictions, average='micro')

	print("Micro-average quality numbers")
	print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
			.format(precision, recall, f1))

	# totprec += precision
	# totrec += recall
	# totF1 += f1

	#MACRO
	precision = precision_score(test_labels, predictions, average='macro')
	recall = recall_score(test_labels, predictions, average='macro')
	f1 = f1_score(test_labels, predictions, average='macro')

	print("Macro-average quality numbers")
	print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
			.format(precision, recall, f1))

	#INDIVIDUAL
	precision = precision_score(test_labels, predictions, average=None)
	recall = recall_score(test_labels, predictions, average=None)
	f1 = f1_score(test_labels, predictions, average=None)

	print("All-Class quality numbers")
	print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
			.format(precision, recall, f1))

	# print "K-fold Micro average:"
	# print("Precision: \n{}, \nRecall: \n{}, \nF1-measure: \n{}"
	#         .format(totprec/10, totrec/10, totF1/10))


	import numpy as np
	exp_train = np.sum(Y_train, axis=0)
	exp_test = np.sum(Y_test, axis=0)
	print "Num of train docs per category:\n", exp_train
	print "Num of test docs per category:\n", exp_test


	#Export to Spreadsheet
	import xlsxwriter

	export = np.column_stack((exp_train, exp_test, f1, precision, recall))
	workbook = xlsxwriter.Workbook('classscores.xlsx')
	worksheet = workbook.add_worksheet()
	worksheet.write(0, 0, "Train(Count)")
	worksheet.write(0, 1, "Test(Count)")
	worksheet.write(0, 2, "F1")
	worksheet.write(0, 3, "Precision")
	worksheet.write(0, 4, "Recall")
	for (x,y), value in np.ndenumerate(export):
	    worksheet.write(x+1, y, value)
	workbook.close()



#-------------------------------------------Main Classes-------------------------------------------

# CNN Implemented based on am implemenation on a keras blog for 20NG datasets
# Link : https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
class KerasBlog_CNN_Classifier:

	def __init__(self, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.0001, batch_size=64, num_epochs=10):
		assert len(filter_sizes) == len(filter_counts)
		assert len(filter_sizes) == len(pool_windows)
		self.filter_sizes = filter_sizes
		self.filter_counts = filter_counts
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.pool_windows = pool_windows
		self.learning_rate = learning_rate
		print("Using Nested CNN with parameters : \nBatch-size : {},  \
											\nFilter-Sizes : {},  \
											\nFilter-Counts : {}, \
											\nLearning Rate : {}, \
											\nPool-Windows : {}".format \
											(self.batch_size, self.filter_sizes, self.filter_counts, self.learning_rate, self.pool_windows) )


	def predict(self, x_train, y_train, x_test, y_test, embeddings, sequence_length, class_count):
		print("Nm of classes : ", class_count)
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, Reshape, MaxPooling1D, Convolution1D, Embedding
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad
		from keras.regularizers import l1, l2

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", int_shape(model_input))
		# model_embedding = Embedding(embeddings.shape[0], 300, input_length=sequence_length, name="embedding")(model_input)
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding", trainable=True)(model_input)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		# model_embedding = Dropout(0.4)(model_embedding)

		conv = model_embedding
		for i in range(len(self.filter_sizes)):
			conv = Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1)(conv)
			print("Convolution shape at loop ", i, " : ", int_shape(conv))
			# conv = MaxPooling1D(pool_size=self.pool_windows[i], strides=int(int_shape(conv)[-2]/self.pool_windows[i]))(conv)
			# conv = MaxPooling1D(pool_size=self.pool_windows[i], strides=self.pool_windows[i])(conv)
			conv = MaxPooling1D(5)(conv)
			print("Max-Pool shape at loop ", i, " : ", int_shape(conv))
		model_conv = Flatten()(conv)

		model_hidden = Dropout(0.8)(model_conv)
		# model_output = Dense(class_count, activation="softmax",
		# 				kernel_regularizer=l2(0.1),
		# 				activity_regularizer=l1(0.1))(model_hidden)
		model_output = Dense(class_count, activation="softmax")(model_hidden)

		model = Model(model_input, model_output)
		# optimizer = Adagrad(lr=self.learning_rate)
		optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
		# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_data=(x_test, y_test), verbose=2, shuffle=True)

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_split=0.1, verbose=2, shuffle=True)

		test_model(model, x_test, y_test, y_train)

		return 0


	def predict_multilabel(self, x_train, y_train, x_test, y_test, embeddings, sequence_length, class_count):
		print("Nm of classes : ", class_count)
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, Reshape, MaxPooling1D, Convolution1D, Embedding
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad
		from keras.regularizers import l1, l2

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", int_shape(model_input))
		# model_embedding = Embedding(embeddings.shape[0], 300, input_length=sequence_length, name="embedding")(model_input)
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding", trainable=True)(model_input)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		# model_embedding = Dropout(0.4)(model_embedding)

		conv = model_embedding
		for i in range(len(self.filter_sizes)):
			conv = Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1)(conv)
			print("Convolution shape at loop ", i, " : ", int_shape(conv))
			# conv = MaxPooling1D(pool_size=self.pool_windows[i], strides=int(int_shape(conv)[-2]/self.pool_windows[i]))(conv)
			# conv = MaxPooling1D(pool_size=self.pool_windows[i], strides=self.pool_windows[i])(conv)
			conv = MaxPooling1D(5)(conv)
			print("Max-Pool shape at loop ", i, " : ", int_shape(conv))
		model_conv = Flatten()(conv)

		model_hidden = Dropout(0.8)(model_conv)
		# model_output = Dense(class_count, activation="softmax",
		# 				kernel_regularizer=l2(0.1),
		# 				activity_regularizer=l1(0.1))(model_hidden)
		model_output = Dense(class_count, activation="sigmoid")(model_hidden)

		model = Model(model_input, model_output)
		# optimizer = Adagrad(lr=self.learning_rate)
		optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		# model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_data=(x_test, y_test), verbose=2, shuffle=True)

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_split=0.1, verbose=2, shuffle=True)

		test_model_multilabel(model, x_test, y_test, y_train)

		return 0






############################################## Extra Functions ##############################################




class CNN_Classifier:

	def __init__(self, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=10):
		assert len(filter_sizes) == len(filter_counts)
		assert len(filter_sizes) == len(pool_windows)
		print("Using CNN with parameters : \nBatch-size : {},  \
											\nFilter-Sizes : {},  \
											\nFilter-Counts : {}, \
											\nPool-Windows : {}, {}", \
											self.batch_size, self.filter_sizes, self.filter_counts, self.pool_windows)
		self.filter_sizes = filter_sizes
		self.filter_counts = filter_counts
		self.pool_windows = pool_windows
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate


	def predict(self, x_train, y_train, x_test, y_test, embeddings, sequence_length, class_count):
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, MaxPooling1D, Convolution1D, Embedding
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", model_input.get_shape)
		model_embedding = Embedding(embeddings.shape[0], 100, input_length=sequence_length, name="embedding")(model_input)
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
		print("Embeddings tensor shape: ", model_embedding.get_shape)
		# model_embedding = Dropout(0.4)(model_embedding)
		conv_blocks = []
		for i in range(len(self.filter_sizes)):
			conv = Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1)(model_embedding)
			conv = MaxPooling1D(pool_size=self.pool_windows[i])(conv)
			conv = Flatten()(conv)
			conv_blocks.append(conv)
		model_conv = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		model_hidden = Dropout(0.3)(model_conv)
		# model_hidden = Dense(2024, activation="relu")(model_hidden)
		# model_hidden = Dropout(0.5)(model_hidden)
		model_hidden = Dense(512, activation="relu")(model_hidden)
		model_hidden = Dropout(0.2)(model_hidden)
		model_hidden = Dense(64, activation="relu")(model_hidden)
		model_output = Dense(class_count, activation="softmax")(model_hidden)
		# model_output = Dense(1, activation="sigmoid")(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate)
		# optimizer = Adagrad(lr=self.learning_rate)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

		print(type(x_train), " ; ", type(y_train))
		print(x_train.shape, ' ; ', x_test.shape)
		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		test_model(model, x_test, y_test, y_train)

		return 0




class Nested_CNN_Classifier:

	def __init__(self, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=15):
		assert len(filter_sizes) == len(filter_counts)
		assert len(filter_sizes) == len(pool_windows)
		self.filter_sizes = filter_sizes
		self.filter_counts = filter_counts
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.pool_windows = pool_windows
		self.learning_rate = learning_rate
		print("Using Nested CNN with parameters : \nBatch-size : {},  \
											\nFilter-Sizes : {},  \
											\nFilter-Counts : {}, \
											\nPool-Windows : {}".format \
											(self.batch_size, self.filter_sizes, self.filter_counts, self.pool_windows) )


	def predict(self, x_train, y_train, x_test, y_test, embeddings, sequence_length, class_count):
		print("Nm of classes : ", class_count)
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, Reshape, MaxPooling1D, Convolution1D, Embedding
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", int_shape(model_input))
		model_embedding = Embedding(embeddings.shape[0], 32, input_length=sequence_length, name="embedding")(model_input)
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding", trainable=True)(model_input)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		# model_embedding = Dropout(0.4)(model_embedding)

		conv = model_embedding
		for i in range(len(self.filter_sizes)):
			conv = Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1)(conv)
			print("Convolution shape at loop ", i, " : ", int_shape(conv))
			# conv = MaxPooling1D(pool_size=self.pool_windows[i], strides=int(int_shape(conv)[-2]/self.pool_windows[i]))(conv)
			conv = MaxPooling1D(pool_size=self.pool_windows[i], strides=self.pool_windows[i])(conv)
			print("Max-Pool shape at loop ", i, " : ", int_shape(conv))
		model_conv = Flatten()(conv)

		model_hidden = Dropout(0.5)(model_conv)
		model_output = Dense(class_count, activation="softmax")(model_hidden)

		model = Model(model_input, model_output)
		# optimizer = Adagrad(lr=self.learning_rate)
		optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
		# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		# model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		test_model(model, x_test, y_test, y_train)

		return 0




class RNN_Classifier:

	def __init__(self, output_size, learning_rate=0.001, batch_size=64, num_epochs=10):
		print("Using RNN with {} Neurons : ", self.output_size)
		self.output_size = output_size
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate


	def predict(self, x_train, y_train, x_test, y_test, embeddings, sequence_length, class_count):
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, Embedding, LSTM
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", model_input.get_shape)
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
		print("Embeddings tensor shape: ", model_embedding.get_shape)
		# model_embedding = Dropout(0.4)(model_embedding)
		model_recurrent  = LSTM(int(embeddings.shape[1]*1.5), activation='tanh', dropout=0.2, recurrent_dropout=0.2)(model_embedding)
		# model_recurrent  = LSTM(embeddings.shape[1], activation='tanh', dropout=0.2)(model_embedding)

		model_hidden = Dense(512, activation="relu")(model_recurrent)
		model_hidden = Dropout(0.5)(model_hidden)
		model_hidden = Dense(64, activation="relu")(model_hidden)
		model_output = Dense(class_count, activation="softmax")(model_hidden)
		# model_output = Dense(1, activation="sigmoid")(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate)
		# optimizer = Adagrad(lr=self.learning_rate)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

		print(type(x_train), " ; ", type(y_train))
		print(x_train.shape, ' ; ', x_test.shape)
		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		test_model(model, x_test, y_test, y_train)

		return 0






class HNN_RR_Classifier:

	def __init__(self, output_size, learning_rate=0.001, batch_size=64, num_epochs=10):
		self.output_size = output_size
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate


	def predict(self, x_train, y_train, x_test, y_test, embeddings, dim2_length, dim3_length, class_count):
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, Embedding, LSTM, TimeDistributed
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad

		input_shape = (dim2_length, dim3_length)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", model_input.get_shape)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
		print("Word Vector (Embeddings) tensor shape: ", model_embedding.get_shape)
		# model_embedding = Dropout(0.4)(model_embedding)

		#######################################################################################################

		model_recurrent_sentences  = TimeDistributed(LSTM(int(embeddings.shape[1]*1.5), activation='tanh', dropout=0.2, recurrent_dropout=0.2))(model_embedding)
		print("Sentence Vector tensor shape: ", model_recurrent_sentences.get_shape)
		model_recurrent  = LSTM(int(embeddings.shape[1]*2), activation='tanh', dropout=0.2, recurrent_dropout=0.2)(model_recurrent_sentences)
		print("Doc Vector tensor shape: ", model_recurrent.get_shape)

		#######################################################################################################

		model_hidden = Dense(512, activation="relu")(model_recurrent)
		model_hidden = Dropout(0.5)(model_hidden)
		model_hidden = Dense(64, activation="relu")(model_hidden)
		model_output = Dense(class_count, activation="softmax")(model_hidden)
		# model_output = Dense(1, activation="sigmoid")(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate)
		# optimizer = Adagrad(lr=self.learning_rate)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

		print(type(x_train), " ; ", type(y_train))
		print(x_train.shape, ' ; ', x_test.shape)
		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		test_model(model, x_test, y_test, y_train)

		return 0






class HNN_CR_Classifier:

	def __init__(self, RNN_output_size, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=10):
		self.output_size = RNN_output_size
		assert len(filter_sizes) == len(filter_counts)
		assert len(filter_sizes) == len(pool_windows)
		self.filter_sizes = filter_sizes
		self.filter_counts = filter_counts
		self.pool_windows = pool_windows
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate


	def predict(self, x_train, y_train, x_test, y_test, embeddings, dim2_length, dim3_length, class_count):
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, MaxPooling1D, Convolution1D, Embedding, LSTM, TimeDistributed
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad

		input_shape = (dim2_length, dim3_length)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", model_input.get_shape)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
		print("Word Vector (Embeddings) tensor shape: ", model_embedding.get_shape)
		# model_embedding = Dropout(0.4)(model_embedding)

		#######################################################################################################

		conv_blocks = []
		for i in range(len(self.filter_sizes)):
			conv = TimeDistributed(Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1))(model_embedding)
			print("Post convolution shape: ", conv.get_shape)
			conv = TimeDistributed(MaxPooling1D(pool_size=self.pool_windows[i]))(conv)
			print("Post Pooling shape: ", conv.get_shape)
			conv = TimeDistributed(Flatten())(conv)
			print("Post Flatten shape: ", conv.get_shape)
			conv_blocks.append(conv)
		model_conv_sentences = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		print("Sentence Vector tensor shape: ", model_conv_sentences.get_shape)
		model_recurrent  = LSTM(1024, activation='tanh', dropout=0.2, recurrent_dropout=0.2)(model_conv_sentences)
		print("Doc Vector tensor shape: ", model_recurrent.get_shape)

		#######################################################################################################

		model_hidden = Dense(512, activation="relu")(model_recurrent)
		model_hidden = Dropout(0.5)(model_hidden)
		model_hidden = Dense(64, activation="relu")(model_hidden)
		model_output = Dense(class_count, activation="softmax")(model_hidden)
		# model_output = Dense(1, activation="sigmoid")(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate)
		# optimizer = Adagrad(lr=self.learning_rate)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

		print(type(x_train), " ; ", type(y_train))
		print(x_train.shape, ' ; ', x_test.shape)
		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		test_model(model, x_test, y_test, y_train)

		return 0





class HNN_RC_Classifier:

	def __init__(self, RNN_output_size, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=10):
		self.output_size = RNN_output_size
		assert len(filter_sizes) == len(filter_counts)
		assert len(filter_sizes) == len(pool_windows)
		self.filter_sizes = filter_sizes
		self.filter_counts = filter_counts
		self.pool_windows = pool_windows
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate


	def predict(self, x_train, y_train, x_test, y_test, embeddings, dim2_length, dim3_length, class_count):
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, MaxPooling1D, Convolution1D, Embedding, LSTM, TimeDistributed
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad

		input_shape = (dim2_length, dim3_length)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", model_input.get_shape)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
		print("Word Vector (Embeddings) tensor shape: ", model_embedding.get_shape)
		# model_embedding = Dropout(0.4)(model_embedding)

		#######################################################################################################


		model_recurrent_sentences  = TimeDistributed(LSTM(int(embeddings.shape[1]*1.5), activation='tanh', dropout=0.2, recurrent_dropout=0.2))(model_embedding)
		print("Sentence Vector tensor shape: ", model_recurrent_sentences.get_shape)

		conv_blocks = []
		for i in range(len(self.filter_sizes)):
			conv = Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1)(model_recurrent_sentences)
			print("Post convolution shape: ", conv.get_shape)
			conv = MaxPooling1D(pool_size=self.pool_windows[i])(conv)
			print("Post Pooling shape: ", conv.get_shape)
			conv = Flatten()(conv)
			print("Post Flatten shape: ", conv.get_shape)
			conv_blocks.append(conv)
		model_conv = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
		print("Doc Vector tensor shape: ", model_conv.get_shape)

		#######################################################################################################

		model_hidden = Dense(512, activation="relu")(model_conv)
		model_hidden = Dropout(0.5)(model_hidden)
		model_hidden = Dense(64, activation="relu")(model_hidden)
		model_output = Dense(class_count, activation="softmax")(model_hidden)
		# model_output = Dense(1, activation="sigmoid")(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate)
		# optimizer = Adagrad(lr=self.learning_rate)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

		print(type(x_train), " ; ", type(y_train))
		print(x_train.shape, ' ; ', x_test.shape)
		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		test_model(model, x_test, y_test, y_train)

		return 0
