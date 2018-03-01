import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
#df = pd.read_csv('eeg_sensor_data_old_data.csv')
#df = pd.read_csv("eeg_sensor_data_user1(two_class).csv")
df = pd.read_csv('eeg_sensor_data_user1_int_mood.csv')
data = df.as_matrix()

X = data[0:data.shape[0],4:14]
#print("X:",X)
Y = data[:,2]

Y = Y.astype(np.int32)
X = X.astype(np.float32)

D = X.shape[1]
#print("D:",D)
M = 10
K = 4     

#plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha = 0.5)
#plt.show()

N = len(Y)
#print("N:",N)
T = np.zeros((N, K))
#print("T:",T)
for i in range(N):
    T[i, Y[i]] = 1


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2

tfX = tf.placeholder(tf.float32, [None, D])
#print("tfX:",tfX)
tfY = tf.placeholder(tf.float32, [None, K])
#print("tfX:",tfY)

W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

logits = forward(tfX, W1, b1, W2, b2)


cost = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(
    labels=tfY,
    logits=logits
  )
) 
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(logits, 1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_op, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
    #print("Pred:",pred)
    if i%1000 == 0:
        print ("Accuracy:",np.mean(Y == pred))
        
'''


df = pd.read_csv('eeg_sensor_data_user1(depressed).csv')
data = df.as_matrix()

X = data[0:data.shape[0],4:14]
#print("X:",X)
Y = data[0:data.shape[0],2]

learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
x = tf.placeholder(tf.float32, [None, X.shape[0]])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([X.shape[0], 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 1], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([1]), name='b2')

hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


Ntrain = int(0.75*len(X))

x_train = X[:Ntrain]
y_train = Y[:Ntrain]


with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(y_train.shape[0] / batch_size)
    for epoch in range(epochs):
         avg_cost = 0
         for i in range(total_batch):
             batch_x, batch_y = X.T, Y.T
             _, c = sess.run([optimiser, cross_entropy], 
                          feed_dict={x: batch_x, y: batch_y})
             avg_cost += c / total_batch
         print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))






         
