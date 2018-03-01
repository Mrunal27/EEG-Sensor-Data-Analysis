from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize


df = pd.read_csv('horizontal_data(updated).csv')
data = df.as_matrix()

num_epochs = 100
#total_series_length = data.shape[1]-1
total_series_length = data.shape[1]-36
print('total_series_length',total_series_length)


truncated_backprop_length  = 10
state_size = 4
num_classes = 5
echo_step = 3
batch_size = 10
num_batches = total_series_length//truncated_backprop_length
num_layers = 3
#print('num_batches',num_batches)



def fetchData():
    x = data[4:14,1:total_series_length]
   # x_normed = x / x.max(axis=0)
    x = normalize(x, axis=1, norm='max')
    #print('x:',x)
    y = np.roll(x, echo_step)
    print('y:',y)
    y[0:echo_step] = 0
    #

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
#print('batchX_placeholder:',batchX_placeholder)

batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
#print('batchY_placeholder:',batchY_placeholder)


init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

state_per_layer_list = tf.unstack(init_state, axis = 0)

rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0,:,:],
                                                       state_per_layer_list[idx][1,:,:])
                                                       for idx in range(num_layers)])

#init_state = tf.placeholder(tf.float32, [batch_size, state_size])
#print('init_state:',init_state)


W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
#print("input_series", inputs_series)
labels_series = tf.unstack(batchY_placeholder, axis=1)
#print("labels_series", labels_series)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_idx in range(1):
        x, y = fetchData()

    print ("x:",x)
    print ("y:",y)
'''

#Forward Pass

stacked_rnn = []

for _ in range(num_layers):
    stacked_rnn.append(tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple = True))

cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple = True)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state=rnn_tuple_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

#plot

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
        
        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")      

    plt.draw()
    plt.pause(0.0001)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    
        
    for epoch_idx in range(num_epochs):
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))
        x,y = fetchData()
        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            #print('start_idx:',start_idx)
            #print('end_idx:',end_idx)
           

            batchX = x[:,start_idx:end_idx]
            #print('batchX:',batchX)
            batchY = y[:,start_idx:end_idx]
            #print('batchY:',batchY)
            #print('current_state (within):',current_state)

                     
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)
                
            if batch_idx%10 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                for predictions in _predictions_series:
                    print("predictions:", predictions)
                plot(loss_list, _predictions_series, batchX, batchY)
            

plt.ioff()
plt.show()

'''




        
    
