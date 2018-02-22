#####################################
## Contains Classification Modules ##
#####################################

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
        import numpy as np

        req_type = type(np.array([]))
        assert type(x_train) == req_type and type(x_test) == req_type
        assert type(y_train) == req_type and type(y_test) == req_type

        from keras.models import Sequential, Model
        from keras.layers import Input, Dense, Dropout, Flatten, MaxPooling1D, Convolution1D, Embedding
        from keras.layers.merge import Concatenate
        from keras.preprocessing import sequence
        from keras.optimizers import Adam
        np.random.seed(123456)

        # input_shape = (sequence_length, embeddings.shape[1])
        input_shape = (sequence_length,)
        model_input = Input(shape=input_shape)
        print("Input tensor shape: ", model_input.get_shape)
        # model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sequence_length, name="embedding")(model_input)
        model_embedding = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], name="embedding")(model_input)
        print("Embeddings tensor shape: ", model_embedding.get_shape)
        model_embedding = Dropout(0.4)(model_embedding)
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
        model_hidden = Dense(1024, activation="relu")(model_hidden)
        model_hidden = Dropout(0.5)(model_hidden)
        model_hidden = Dense(256, activation="relu")(model_hidden)
        model_hidden = Dropout(0.2)(model_hidden)
        model_hidden = Dense(64, activation="relu")(model_hidden)
        model_output = Dense(class_count, activation="softmax")(model_hidden)
        # model_output = Dense(1, activation="sigmoid")(model_hidden)

        model = Model(model_input, model_output)
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(type(x_train), " ; ", type(y_train))
        print(x_train.shape, ' ; ', x_test.shape)
        # model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
        #   validation_data=(x_test, y_test), verbose=2, shuffle=True)

        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
          validation_split=0.2, verbose=2, shuffle=True)

        return 0
