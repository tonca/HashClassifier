# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np

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


BATCH_SIZE = 32 
TEST_SIZE = 100

gen_train = genh.hash_generator(BATCH_SIZE)
gen_test = genh.hash_generator(TEST_SIZE)
# create the model
model = Sequential()
model.add(Dense(32, input_shape=(None,256), activation='relu'))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(2, activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit_generator(gen_train, steps_per_epoch=BATCH_SIZE, epochs=10)

# Final evaluation of the model
scores = model.evaluate_generator(gen_test, 100)
print("Accuracy: %.10f%%" % (scores[1]*100))