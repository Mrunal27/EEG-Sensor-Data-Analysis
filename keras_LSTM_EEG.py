from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

#from sklearn.utils import plot_model

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import losses



#df = pd.read_csv("eeg_sensor_data_user1(two_class).csv",usecols=[4,5,6,7,8,9,10,11,12,13,14], engine='python', skipfooter=3)
df = pd.read_csv("eeg_sensor_data_user1(updated).csv")
test_df = pd.read_csv("eeg_sensor_data_user1(two_class).csv")

data = df.as_matrix()
test_data = df.as_matrix()

#plt.plot(df)
#plt.show()


x = data[0: data.shape[0], 4:14]
print("x:", x.shape)

y = data[0:data.shape[0],2]
print("y:", y.shape)




data_dim = x.shape[1]
timesteps = 1
#num_classes = 2 Happy, Sad
num_classes = 3  #Happy, Sad, Drowsy


encoder = LabelEncoder()
encoder.fit(y)

#print("encoder:", encoder)

encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
print("encoder:", encoded_Y)

y = keras.utils.to_categorical(encoded_Y, num_classes)
print("dymmy:", y.shape)

Ntrain = int(0.7*len(x))

'''
#Training Data split into 5 classes
x_train = x[:Ntrain]
print("x_train.shape", x_train.shape)
y_train = y[:Ntrain]
print("y_train.shape", y_train.shape)
'''

x_train = x
print("x_train.shape", x_train.shape)
y_train = y
print("y_train.shape", y_train.shape)


# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(50, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(100, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(1))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

'''
x_test = x[Ntrain:]
y_test = y[Ntrain:]
#print("y_test:",y_test)
#print("before reshape", x_train)
'''

x_train = np.reshape(x_train, (x_train.shape[0], timesteps, x_train.shape[1]))

x_test = test_data[0: test_data.shape[0], 4:14]
y_test = data[0:test_data.shape[0],2]
x_test = np.reshape(x_test, (x_test.shape[0], timesteps, x_test.shape[1]))

encoded_Y = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
print("encoder:", encoded_Y)

y_test = keras.utils.to_categorical(encoded_Y, num_classes)


#model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))
model.fit(x_train, y_train, batch_size=100, epochs=10)
scores = model.evaluate(x_test, y_test, verbose=0,batch_size=100)
#print(scores)
print("Accuracy: %.2f%%" % (scores[1]*100))


predictions = model.predict_on_batch(x_test)
# round predictions
rounded = [round(x_test[0],0) for x_test in predictions]
print(predictions)

#print("MSE:", losses.mean_squared_error(y_test, predictions))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_test)
plt.plot(predictions)
plt.show()

#plot_model(model, to_file='model.png')
#model.pyplot()














