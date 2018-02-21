
################################### My Part ###################################

from __future__ import print_function
import tensorflow as tf
tf.set_random_seed(100)
import random
random.seed(100)
import numpy as np
np.random.seed(100)

from keras.preprocessing.sequence import pad_sequences
from Create_Data_Model import processed_data, tagMatrix2Embeddings
from Other_Utils import prob2Onehot3D, pad_sequences_3D
data = processed_data()


trainX, word_embeddings, trainY, max_sentence_length, POS_labels = data.get_Data_Embeddings()
#POS_vectors, POS_embeddings, _ = tagMatrix2Embeddings(POS_labels)
len_vocab = len(word_embeddings)
del data


x_train = pad_sequences(trainX, maxlen=max_sentence_length, value=0)
#POS_vectors = pad_sequences(POS_vectors, maxlen=maxLen, value=0)
y_train_hot = pad_sequences_3D(trainY, maxlen=max_sentence_length, value=[0,0,1])

x_test = x_train[int(0.3*len(trainY)):]
#test_POS_vectors = POS_vectors[int(0.3*len(trainY)):]
testY = y_train_hot[int(0.3*len(trainY)):]





################################### Alapanda Part ###################################

import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Conv1D, LSTM, Bidirectional, Input, merge, TimeDistributed, Concatenate


max_features = len_vocab#vocab size
batch_size = 40     #batch size
maxlen = max_sentence_length       #max tweet_characterized length
hidden = 60     #size of hidden layer
nb_classes = 3
#filter_sizes = [2,3,4]
num_filters = 30
embd_len = len(word_embeddings[0])


sequence = Input(shape=(maxlen,), dtype='int32')
embedded = Embedding(max_features, embd_len, input_length=maxlen, weights=[word_embeddings])(sequence)
print(np.shape(embedded))
embedded1= Conv1D(filters=30, kernel_size=[3], strides=1, padding='same')(embedded)
print(np.shape(embedded1))
embedded2= Conv1D(filters=20, kernel_size=[4], strides=1, padding='same')(embedded1)
print(np.shape(embedded2))
# forwards = LSTM(hidden, return_sequences=True)(embedded2)
# backwards = LSTM(hidden, return_sequences=True, go_backwards=True)(embedded2)
# merged = Concatenate(axis=-1)([forwards, backwards])
merged = Bidirectional(LSTM(hidden, return_sequences=True), merge_mode='concat')(embedded2)
after_dp = Dropout(0.5)(merged)
# after_dp = Bidirectional(LSTM(hidden, dropout=0.5, recurrent_dropout=0.2, return_sequences=True), merge_mode='concat')(embedded2)
print(np.shape(after_dp))
output = TimeDistributed(Dense(nb_classes, activation='softmax'))(after_dp)
print(np.shape(output))
model = Model(inputs=sequence, outputs=output)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
y_train_hot=np.array(y_train_hot)
model.fit(x_train, y_train_hot,batch_size=batch_size,epochs=15,validation_split=0.2)
y_pred=model.predict(x_test)





################################### My Part ###################################

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
