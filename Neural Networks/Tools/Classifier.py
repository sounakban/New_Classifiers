#####################################
## Contains Classification Modules ##
#####################################

#-----------------------------------Common Functions & Imports-------------------------------------

import numpy as np
np.random.seed(1337)
from tensorflow import set_random_seed
set_random_seed(2017)
from keras.backend import int_shape
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]


def plot_AccLoss_Curve(history):
    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    # summarize history for Accuracy and Loss
    print("Type: ", type(history.history['val_acc']))
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['val_loss'])
    plt.title('Accuracy-Loss Graph')
    # plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    # plt.show()
    fig = plt.figure()
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


class CNN_Classifier:

    def __init__(self, filter_sizes=[], filter_counts=[], pool_windows=[], learning_rate=0.001, batch_size=64, num_epochs=10):
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
        # model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
        model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
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
            conv = MaxPooling1D(pool_size=self.pool_windows[i])(conv)
            conv = Flatten()(conv)
            # conv = Reshape((-1,))(conv)
            conv_blocks.append(conv)
        model_conv = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        model_hidden = Dropout(0.5)(model_conv)
        model_hidden = Dense(int(int_shape(model_hidden)[-1]*2), activation="relu")(model_hidden)
        model_hidden = Dropout(0.5)(model_hidden)
        model_hidden = Dense(1024, activation="relu")(model_hidden)
        model_hidden = Dropout(0.6)(model_hidden)
        model_hidden = Dense(64, activation="relu")(model_hidden)
        model_output = Dense(class_count, activation="softmax")(model_hidden)
        # model_output = Dense(1, activation="sigmoid")(model_hidden)

        model = Model(model_input, model_output)
        optimizer = Adam(lr=self.learning_rate)
        # optimizer = Adagrad(lr=self.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        # model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
        #   validation_data=(x_test, y_test), verbose=2, shuffle=True)

        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
          validation_split=0.2, verbose=2, shuffle=True)

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
