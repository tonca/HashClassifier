# LSTM and CNN for sequence classification in the IMDB dataset
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils


import generate_hashes as genh


X_train, y_train = genh.generate_hashes(5000)
X_test, y_test = genh.generate_hashes(1000)
y_train = np_utils.to_categorical(y_train).reshape((-1,6))
y_test = np_utils.to_categorical(y_test).reshape((-1,6))

# create the model
model = Sequential()
model.add(Dense(32, input_shape=(None,256), activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(6, activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, epochs=3)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.5f%%" % (scores[1]*100))