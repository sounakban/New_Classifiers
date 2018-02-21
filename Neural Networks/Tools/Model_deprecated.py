
def train_Classify(trainX, trainY, testX, testY, nb_classes, max_features, maxlen, word_embeddings, ):

    ################################### Data Preparation ###################################

    from __future__ import print_function
    import tensorflow as tf
    tf.set_random_seed(100)
    import random
    random.seed(100)
    import numpy as np
    np.random.seed(100)

    from keras.preprocessing.sequence import pad_sequences
    from Other_Utils import prob2Onehot3D, pad_sequences_3D

    x_train = pad_sequences(trainX, maxlen=max_sentence_length, value=0)
    y_train_hot = pad_sequences_3D(trainY, maxlen=max_sentence_length, value=[0,0,1])

    x_test = x_train[int(0.3*len(trainY)):]
    testY = y_train_hot[int(0.3*len(trainY)):]


    ################################### Network ###################################

    import numpy as np
    import keras
    from keras.preprocessing import sequence
    from keras.models import Model
    from keras.layers import Dense, Dropout, Embedding, Conv1D, LSTM, Bidirectional, Input, merge, TimeDistributed, Concatenate


    max_features = len_vocab            #vocab size
    embd_len = len(word_embeddings[0])  #Size of each embedding
    nb_classes = 3
    maxlen = max_sentence_length        #max tweet_characterized length

    ## HyperParameters ##
    batch_size = 50                     #batch size
    hidden_layer_size = [1024]          #size of hidden layer
    # filter_sizes = [2,3,4]
    # num_filters = [30,40,50]


    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(max_features, embd_len, input_length=maxlen, weights=[word_embeddings])(sequence)
    print(np.shape(embedded))
    convolution_out = Conv1D(filters=30, kernel_size=[3], strides=1, padding='same')(embedded)
    print(np.shape(convolution))
    convolution_out = Conv1D(filters=20, kernel_size=[4], strides=1, padding='same')(convolution)
    print(np.shape(convolution))
    hidden_out = Dense(hidden_layer_size[0], activation='softmax')(convolution_out)
    print(np.shape(output))
    output = Dense(nb_classes, activation='softmax')(hidden_out)
    print(np.shape(output))
    model = Model(inputs=sequence, outputs=output)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    y_train_hot=np.array(y_train_hot)
    model.fit(x_train, y_train_hot,batch_size=batch_size,epochs=15,validation_split=0.2)
    y_pred=model.predict(x_test)





################################### Evaluate ###################################

predictions = prob2Onehot3D(y_pred)
#print("Predictions : ", list(predictions[10]))

##Calculate F1 Score
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(predictions.shape[0]):
    for j in range(predictions.shape[1]):
        if list(testY[i][j]) == [0,1,0] or list(testY[i][j]) == [1,0,0]:
            if list(predictions[i][j]) == [0,1,0] or list(predictions[i][j]) == [1,0,0]:
                tp += 1
            else:
                fn += 1
        else:
            if list(predictions[i][j]) == [0,1,0] or list(predictions[i][j]) == [1,0,0]:
                fp += 1
            else:
                tn += 1

print(predictions.shape)
print(x_test.shape)
print(testY.shape)
print("Tru-Pos : ", tp)
print("Tru-Neg : ", tn)
print("Fals-Pos : ", fp)
print("Fals-Neg : ", fn)

pr = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = 2*((pr*rec)/(pr+rec))
print("Precision : ", pr)
print("Recall : ", rec)
print("F1 : ", f1)
