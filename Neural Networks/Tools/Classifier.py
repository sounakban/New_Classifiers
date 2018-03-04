#####################################
## Contains Classification Modules ##
#####################################

#-----------------------------------Common Functions & Imports-------------------------------------

import numpy as np
np.random.seed(123456)

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
        # model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
        model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
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

        model_hidden = Dropout(0.5)(model_conv)
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

        test_model(model, x_test, y_test)

        return 0






class RNN_Classifier:

    def __init__(self, output_size, learning_rate=0.001, batch_size=64, num_epochs=10):
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

        test_model(model, x_test, y_test)

        return 0
