import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense


traindata=np.load('traindata.npy')
traindata /=  traindata.sum(axis=1)[:,np.newaxis]
testdata=np.load('testdata.npy')
testdata /=  testdata.sum(axis=1)[:,np.newaxis]
trainlabel=np.load('trainlabel.npy')
testlabel=np.load('testlabel.npy')



model = Sequential([
                    Dense(32, activation='relu', input_shape=(39,)),
                    Dense(50, activation='relu'),
                    Dense(1, activation='sigmoid'),
                    ])



sgd = optimizers.Adam(lr=0.01)
model.compile(optimizer=sgd,
                loss='binary_crossentropy',
              metrics=['accuracy'])
              
model.fit(traindata,trainlabel,epochs=200)


test_loss, test_acc = model.evaluate(testdata, testlabel)

print('Test accuracy:', test_acc)
model.save('JanPerformance.h5')