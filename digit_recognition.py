import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import sgd
import numpy as np
from keras.layers import Activation

batch_size=128
classes=10
epochs=6

length=28
breadth=28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print( x_train.shape[0])
x_train=x_train.reshape(x_train.shape[0],length,breadth,1).astype('float32')
x_test=x_test.reshape(x_test.shape[0],length,breadth,1).astype('float32')
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

#X_train = x_train / 255
#X_test = x_test / 255
model=Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(classes,activation='softmax'))
opt = keras.optimizers.adam(lr=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.)

#rmsprop
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_train.astype('float32')
x_train /= 255
x_test /= 255
model.summary()
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_split=0.1)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")