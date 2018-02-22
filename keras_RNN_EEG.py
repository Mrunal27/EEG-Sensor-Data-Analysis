from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
import keras



df = pd.read_csv("eeg_sensor_data_old_data.csv")
data = df.as_matrix()



x = data[0:data.shape[0], 4:14]
y = data[:,2]

#x, y = shuffle(x, y)
Ntrain = int(0.7*len(x))

x_train = x[:Ntrain]
y_train = y[:Ntrain]
y_train = keras.utils.to_categorical(y_train, num_classes=5)


data_dim = x_train.shape[1]
timesteps = 1
num_classes = 5
batch_size = 100
print("batch_size", batch_size)

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful = True,
               batch_input_shape=(batch_size,timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True, stateful = True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, stateful = True))  # return a single vector of dimension 32
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

x_test = x[Ntrain:]
y_test = y[Ntrain:]
y_test = keras.utils.to_categorical(y_test, num_classes=5)


x_train = np.reshape(x_train, (x_train.shape[0], timesteps, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], timesteps, x_test.shape[1]))



model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))











