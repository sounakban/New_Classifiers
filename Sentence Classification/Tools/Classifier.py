#####################################
## Contains Classification Modules ##
#####################################

#-----------------------------------Common Functions & Imports-------------------------------------

import numpy as np
# np.random.seed(1337)
from tensorflow import set_random_seed
# set_random_seed(2017)
from keras.backend import int_shape
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]


def plot_AccLoss_Curve(history):
	import matplotlib
	matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
	import matplotlib.pyplot as plt
	# summarize history for Accuracy and Loss
	plt.plot(history.history['val_acc'])
	plt.plot(history.history['val_loss'])
	plt.title('Accuracy-Loss Graph')
	# plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['Accuracy', 'Loss'], loc='upper left')
	# plt.show()
	fig = plt.gcf()
	fig.savefig("test1.png")


def test_model(model, X_test, Y_test):
	from keras.utils import to_categorical
	#Get Class of max predicted value
	predictions = model.predict(X_test).argmax(axis=-1)
	#Convert to binary category matrix with int type
	predictions = np.array(to_categorical(predictions), dtype=np.int16)

	from sklearn.metrics import f1_score, precision_score, recall_score
	test_labels = Y_test

	#MICRO
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




#-------------------------------------------Main Classes-------------------------------------------


class Nested_CNN_Classifier:

	def __init__(self, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=10):
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

		model_hidden = Dropout(0.3)(model_conv)
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

		test_model(model, x_test, y_test)

		return 0






# Vanilla CNN implementation, Not Customized with anything
class CNN_Classifier:

	def __init__(self, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=20):
		assert len(filter_sizes) == len(filter_counts)
		assert len(filter_sizes) == len(pool_windows)
		self.filter_sizes = filter_sizes
		self.filter_counts = filter_counts
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.pool_windows = pool_windows
		self.learning_rate = learning_rate
		print("Using CNN with parameters : \nBatch-size : {},  \
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
		from keras.optimizers import Adam, Adagrad, Adadelta
		from keras.regularizers import l2

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", int_shape(model_input))
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
		# model_embedding = Embedding(embeddings.shape[0], 100, input_length=sequence_length, name="embedding")(model_input)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding", trainable=True)(model_input)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		# model_embedding = Dropout(0.4)(model_embedding)
		conv_blocks = []
		for i in range(len(self.filter_sizes)):
			conv = Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1)(model_embedding)
			# conv = MaxPooling1D(pool_size=self.pool_windows[i])(conv)
			conv = MaxPooling1D(pool_size=sequence_length-self.filter_sizes[i])(conv)
			print("Pool shape: ", int_shape(conv))
			conv = Flatten()(conv)
			conv_blocks.append(conv)
		model_conv = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		model_hidden = Dropout(0.5)(model_conv)
		model_output = Dense(class_count, activation="softmax", kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1))(model_hidden)

		model = Model(model_input, model_output)
		# optimizer = Adagrad(lr=self.learning_rate)
		optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
		# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		# model.compile(loss="binary_crossentropy", optimizer=optimizer, metrself.ics=["accuracy"])

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		score, acc = model.evaluate(x_test, y_test,
							batch_size=self.batch_size)
		print('Test score:', score)
		print('Test accuracy:', acc)

		test_model(model, x_test, y_test)

		return 0






 # Parameters have been tuned based on the paper : Convolutional Neural Networks for Sentence Classification
 # Link : https://arxiv.org/pdf/1408.5882.pdf
class CNN1_Classifier:

	def __init__(self, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=20):
		assert len(filter_sizes) == len(filter_counts)
		assert len(filter_sizes) == len(pool_windows)
		self.filter_sizes = filter_sizes
		self.filter_counts = filter_counts
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.pool_windows = pool_windows
		self.learning_rate = learning_rate
		print("Using CNN with parameters : \nBatch-size : {},  \
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
		from keras.optimizers import Adam, Adagrad, Adadelta
		from keras.regularizers import l2

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", int_shape(model_input))
		# model_embedding = Embedding(embeddings.shape[0], 100, input_length=sequence_length, name="embedding")(model_input)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding", trainable=True)(model_input)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		conv_blocks = []
		for i in range(len(self.filter_sizes)):
			conv = Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1)(model_embedding)
			conv = MaxPooling1D(pool_size=sequence_length-self.filter_sizes[i])(conv)
			print("Pool shape: ", int_shape(conv))
			conv = Flatten()(conv)
			conv_blocks.append(conv)
		model_conv = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		model_hidden = Dropout(0.5)(model_conv)
		model_output = Dense(class_count, activation="softmax", kernel_regularizer=l2(3), bias_regularizer=l2(3))(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		score, acc = model.evaluate(x_test, y_test,
							batch_size=self.batch_size)
		print('\nTest score:', score)
		print('Test accuracy:', acc, '\n')


		test_model(model, x_test, y_test)

		return 0







 # Parameters have been tuned based on the implementation in the link
 # Link : https://github.com/amitvpatel06/Twitter-Deep-Learning
class CNN2_Classifier:

	def __init__(self, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=20):
		assert len(filter_sizes) == len(filter_counts)
		assert len(filter_sizes) == len(pool_windows)
		self.filter_sizes = filter_sizes
		self.filter_counts = filter_counts
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.pool_windows = pool_windows
		self.learning_rate = learning_rate
		print("Using CNN with parameters : \nBatch-size : {},  \
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
		from keras.optimizers import Adam, Adagrad, Adadelta
		from keras.regularizers import l2

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", int_shape(model_input))
		model_embedding = Embedding(embeddings.shape[0], 100, input_length=sequence_length, name="embedding")(model_input)
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding", trainable=True)(model_input)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		conv_blocks = []
		for i in range(len(self.filter_sizes)):
			conv = Convolution1D(filters=self.filter_counts[i],
								 kernel_size=self.filter_sizes[i],
								 padding="valid",
								 activation="relu",
								 use_bias=False,
								 strides=1)(model_embedding)
			conv = MaxPooling1D(pool_size=sequence_length-self.filter_sizes[i])(conv)
			print("Pool shape: ", int_shape(conv))
			conv = Flatten()(conv)
			conv_blocks.append(conv)
		model_conv = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

		model_hidden = Dropout(0.5)(model_conv)
		model_output = Dense(class_count, activation="softmax", kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1))(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		# model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
		model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		score, acc = model.evaluate(x_test, y_test,
							batch_size=self.batch_size)
		print('\nTest score:', score)
		print('Test accuracy:', acc, '\n')


		test_model(model, x_test, y_test)

		return 0




class Stacked_BiLSTM_Classifier:

	def __init__(self, output_size, learning_rate=0.001, batch_size=64, num_epochs=10):
		self.output_size = output_size
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		print("Using BiDirectional-RNN with parameters : \nBatch-size : {},  \
											\nLearning-Rate : {},  \
											\nNeurons : {}".format \
											(self.batch_size, self.learning_rate, self.output_size) )


	def predict(self, x_train, y_train, x_test, y_test, embeddings, sequence_length, class_count):
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, Embedding, LSTM, Bidirectional, LSTMCell, StackedRNNCells, RNN
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad
		from keras.regularizers import l2

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", int_shape(model_input))
		model_embedding = Embedding(embeddings.shape[0], 100, input_length=sequence_length, name="embedding")(model_input)
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		# model_recurrent = Bidirectional(LSTM(embeddings.shape[1], activation='relu', dropout=0.2))(model_embedding)
		#####################################################################################################################################

		# cells_forward = [LSTMCell(units=self.output_size), LSTMCell(units=self.output_size), LSTMCell(units=self.output_size)]
		# cells_backward = [LSTMCell(units=self.output_size), LSTMCell(units=self.output_size), LSTMCell(units=self.output_size)]
		cells_forward = [LSTMCell(units=self.output_size)] * 3
		cells_backward = [LSTMCell(units=self.output_size)] * 3
		# LSTM_forward = RNN(cells_forward, go_backwards=False)(model_embedding)
		# LSTM_backward = RNN(cells_backward, go_backwards=True)(model_embedding)

		cells_forward_stacked = StackedRNNCells(cells_forward)
		cells_backward_stacked = StackedRNNCells(cells_backward)
		LSTM_forward = RNN(cells_forward_stacked, go_backwards=False)(model_embedding)
		LSTM_backward = RNN(cells_backward_stacked, go_backwards=True)(model_embedding)

		model_recurrent = Concatenate(axis=-1)([LSTM_forward, LSTM_backward])
		# model_recurrent = Bidirectional(cells_forward)(model_embedding)

		######################################################################################################################################
		model_hidden = Dropout(0.5)(model_recurrent)
		model_output = Dense(class_count, activation="softmax", kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1))(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate)
		# optimizer = Adagrad(lr=self.learning_rate)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		score, acc = model.evaluate(x_test, y_test,
							batch_size=self.batch_size)
		print('\nTest score:', score)
		print('Test accuracy:', acc, '\n')

		test_model(model, x_test, y_test)

		return 0





class BDRNN_Classifier:

	def __init__(self, output_size, learning_rate=0.001, batch_size=64, num_epochs=10):
		self.output_size = output_size
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		print("Using BiDirectional-RNN with parameters : \nBatch-size : {},  \
											\nLearning-Rate : {},  \
											\nNeurons : {}".format \
											(self.batch_size, self.learning_rate, self.output_size) )


	def predict(self, x_train, y_train, x_test, y_test, embeddings, sequence_length, class_count):
		req_type = type(np.array([]))
		assert type(x_train) == req_type and type(x_test) == req_type
		assert type(y_train) == req_type and type(y_test) == req_type

		from keras.models import Model
		from keras.layers import Input, Dense, Dropout, Flatten, Embedding, LSTM, Bidirectional
		from keras.layers.merge import Concatenate
		from keras.optimizers import Adam, Adagrad

		input_shape = (sequence_length,)
		model_input = Input(shape=input_shape)
		print("Input tensor shape: ", int_shape(model_input))
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		model_recurrent = Bidirectional(LSTM(embeddings.shape[1], activation='relu', dropout=0.2))(model_embedding)

		model_hidden = Dense(int(int_shape(model_recurrent)[-1]/2), activation="relu")(model_recurrent)
		model_hidden = Dropout(0.5)(model_hidden)
		model_hidden = Dense(256, activation="relu")(model_hidden)
		model_hidden = Dropout(0.8)(model_hidden)
		model_hidden = Dense(128, activation="relu")(model_hidden)
		model_hidden = Dropout(0.6)(model_hidden)
		model_hidden = Dense(64, activation="relu")(model_hidden)
		model_output = Dense(class_count, activation="softmax")(model_hidden)
		# model_output = Dense(1, activation="sigmoid")(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate)
		# optimizer = Adagrad(lr=self.learning_rate)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)

		test_model(model, x_test, y_test)

		return 0



class RNN_Classifier:

	def __init__(self, output_size, learning_rate=0.001, batch_size=64, num_epochs=10):
		self.output_size = output_size
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		print("Using RNN with parameters : \nBatch-size : {},  \
											\nLearning-Rate : {},  \
											\nNeurons : {}".format \
											(self.batch_size, self.learning_rate, self.output_size) )


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
		print("Input tensor shape: ", int_shape(model_input))
		# model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
		model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
		# print("Embeddings tensor shape: ", model_embedding.get_shape)
		print("Embeddings tensor shape: ", int_shape(model_embedding))
		# model_embedding = Dropout(0.4)(model_embedding)
		model_recurrent  = LSTM(int(embeddings.shape[1]*1.5), activation='relu', dropout=0.5, recurrent_dropout=0.2)(model_embedding)
		# model_recurrent  = LSTM(embeddings.shape[1], activation='relu', dropout=0.2)(model_embedding)

		model_hidden = Dense(int(int_shape(model_recurrent)[-1]/2), activation="relu")(model_recurrent)
		model_hidden = Dropout(0.5)(model_hidden)
		model_hidden = Dense(512, activation="relu")(model_hidden)
		model_hidden = Dropout(0.6)(model_hidden)
		# model_hidden = Dense(128, activation="relu")(model_hidden)
		# model_hidden = Dropout(0.6)(model_hidden)
		model_hidden = Dense(64, activation="relu")(model_hidden)
		model_output = Dense(class_count, activation="softmax")(model_hidden)
		# model_output = Dense(1, activation="sigmoid")(model_hidden)

		model = Model(model_input, model_output)
		optimizer = Adam(lr=self.learning_rate)
		# optimizer = Adagrad(lr=self.learning_rate)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
		# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

		# model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		#   validation_data=(x_test, y_test), verbose=2, shuffle=True)

		history = model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
		  validation_split=0.2, verbose=2, shuffle=True)
		plot_AccLoss_Curve(history)

		test_model(model, x_test, y_test)

		return 0
