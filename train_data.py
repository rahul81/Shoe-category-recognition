# importing the required libraries

import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python import keras
from sklearn.utils import shuffle

# loading the numpy files with image and label data

x= np.load('.../Image_numpy_file.npy')
y= np.load('.../labels_numpy_file.npy')



x = x/255.0

# number of unique categories in data
num_classes = 6



# reshaping the data to feed in the CNN
# the four parameters are no. of images, width,height,channels

x = x.reshape(-1,128,128,1)


train_x=x[:]
train_y=y[:]

train_x,train_y=shuffle(train_x,train_y)



# model architechture

model = Sequential()

# Input layer 1

model.add(Conv2D(128, kernel_size=(7, 7),
                 activation='relu',
                 input_shape=(128,128,1)))
model.add(MaxPooling2D(pool_size=(3,3)))

# layer 2
model.add(Conv2D(256, (7, 7), activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

# flattening layer
model.add(Flatten())

# Dense layer
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

# Dense output layer
model.add(Dense(num_classes, activation='softmax'))



# compiling the model with apprpriate metrics, optimizer and loss function
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# training the model
model.fit(train_x, train_y, batch_size=5, epochs=50, validation_split=0.2)

# saving the trained model
model.save('.../filename.model')
