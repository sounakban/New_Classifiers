###### This module will perform trigger detection #######

import tensorflow as tf
tf.set_random_seed(100)
import random
random.seed(100)
import numpy as np
np.random.seed(100)

from Create_Data_Model import processed_data, tagMatrix2Embeddings
from Other_Utils import prob2Onehot3D, pad_sequences_3D
data = processed_data()

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Bidirectional, Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

#Get data
trainX, word_embeddings, trainY, maxLen, POS_labels = data.get_Data_Embeddings()
POS_vectors, POS_embeddings, _ = tagMatrix2Embeddings(POS_labels)
del data

# print("TrainX : ", len(trainX))
# print("TrainY : ", len(trainY))
# print("Word Embd : ", len(word_embeddings))
# print("POS labels: ", len(POS_labels))
# print("POS vecs: ", len(POS_vectors))
# print("POS Embd: ", POS_embeddings.shape)
# print("Max Len : ", maxLen)

# Data preprocessing
## Sequence padding
trainX = pad_sequences(trainX, maxlen=maxLen, value=0)
POS_vectors = pad_sequences(POS_vectors, maxlen=maxLen, value=0)
trainY = pad_sequences_3D(trainY, maxlen=maxLen, value=[0,0,1])

# tot = 0
# for a in trainY:
#     for b in a:
#         if list(b) == [1,0,0] or list(b) == [0,1,0]:
#             tot += 1
# print("Total positive results : ", tot)

# Defining the Network
print("Beginning neural network")

## Defining vectors and embeddings
word_inp = Input(shape=(maxLen,))
word_embed_layer = Embedding(len(word_embeddings), len(word_embeddings[0]), weights=[word_embeddings], input_length=maxLen)(word_inp)
print("Shape, word embd: ", np.shape(word_embed_layer))
POS_inp = Input(shape=(maxLen,))
POS_embed_layer = Embedding(len(POS_embeddings), len(POS_embeddings[0]), weights=[POS_embeddings], input_length=maxLen)(POS_inp)
# POS_embed_layer.set_weights(POS_embeddings)
print("Shape, POS embd: ", np.shape(POS_embed_layer))

## Combine Embeddings
embed_layer = Concatenate(axis=-1)([word_embed_layer, POS_embed_layer])
# embed_layer = word_embed_layer
print("Shape, total embd: ", np.shape(embed_layer))

## Layer Operations
# print(net.get_shape().as_list())
# seq = Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.2, return_sequences=True), merge_mode='concat')(embed_layer)
# seq = Bidirectional(LSTM(256, dropout=0.5, return_sequences=True), merge_mode='concat')(embed_layer)
#seq = Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), merge_mode='concat')(embed_layer)
# print("Shape of Bi-LSTM output : ", np.shape(seq))


forwards = LSTM(512, return_sequences=True, activation='tanh', recurrent_dropout=0.2)(embed_layer)
backwards = LSTM(512, return_sequences=True, go_backwards=True, activation='tanh', recurrent_dropout=0.2)(embed_layer)
seq = Concatenate(axis=-1)([forwards, backwards])


print(np.shape(seq))
seq = Dropout(0.5)(seq)
seq = Concatenate(axis=-1)([seq, POS_embed_layer])
mlp = seq
mlp = TimeDistributed(Dense(3, activation='softmax'))(mlp)
model = Model(inputs=[word_inp, POS_inp], outputs=mlp)
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

testX = trainX[int(0.3*len(trainY)):]
test_POS_vectors = POS_vectors[int(0.3*len(trainY)):]
testY = trainY[int(0.3*len(trainY)):]


# Training
model.fit([trainX, POS_vectors], trainY, epochs=20, validation_split=0.2, batch_size=32, shuffle=True)
predictions = model.predict([testX, test_POS_vectors])
predictions = prob2Onehot3D(predictions)
print("Predictions : ", list(predictions[10]))


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
print(testX.shape)
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
