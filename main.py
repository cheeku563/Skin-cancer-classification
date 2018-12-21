from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf 
from tensorflow.contrib.layers import flatten
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from functools import partial

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("",one_hot= True)
X_train= mnist.train.images
Y_train=mnist.train.labels

n_epochs = 1000
batch_size=15

learning_rate=0.1

n_inputs = 28*28
n_hidden_1 = 300
n_hidden_2 = 144
n_hidden_3 = 300
n_outputs=28*28

l2 = 0.001
n = len(mnist.train.images)
n=500
X = tf.placeholder(tf.float32,shape=(None,784))
Y = tf.placeholder(tf.float32,shape = (None,10))
def get_weights():
	return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('kernel:0')]

def get_biases():
	return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('bias:0')]


acti = tf.nn.elu
he_init = tf.contrib.layers.variance_scaling_initializer()
l2_reg = tf.contrib.layers.l2_regularizer(l2)


output_h1 = tf.layers.dense(X,n_hidden_1,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h2 = tf.layers.dense(output_h1,n_hidden_2,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h3 = tf.layers.dense(output_h2,n_hidden_3,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output = tf.layers.dense(output_h3,n_inputs,activation = None,kernel_initializer= he_init,kernel_regularizer = l2_reg)

reconstruction_loss = tf.reduce_mean(tf.square(output-X))
#reg_loss = l2_reg(get_weights()[0])+l2_reg(get_weights()[1])+l2_reg(get_weights()[2])+l2_reg4(get_weights()[3])
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add(reconstruction_loss,reg_loss)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

def plot_image(image,shape=[28,28]):
	plt.imshow(image.reshape(shape),cmap="Greys",interpolation = "nearest")
	plt.axis("off")
def plot_image_h2(image,shape=[15,10]):
	plt.imshow(image.reshape(shape),cmap="Greys",interpolation = "nearest")
	plt.axis("off")
	

#nn = tf.layers.dense(output_h2,150,activation=tf.nn.relu,kernel_initializer=he_init,kernel_regularizer=l2_reg)

#nn2 = tf.layers.dense(nn,100,activation=tf.nn.relu,kernel_initializer=he_init,kernel_regularizer=l2_reg)
#logits = tf.layers.dense(nn2,10,activation=None,kernel_initializer=he_init,kernel_regularizer=l2_reg)
#output_f = tf.sigmoid(logits)



output_h2 = tf.reshape(output_h2, shape =(-1,12,12,1))
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
output_h2 = tf.pad(output_h2, [[0, 0], [10, 10], [10, 10], [0, 0]], mode="CONSTANT")
    # TODO: Define the LeNet architecture.
    # Return the result of the last fully connected layer.

    #Convolution layer 1. The output shape should be 28x28x6.
conv1_filter_w = tf.Variable(tf.truncated_normal([5,5,1,6]))
conv1_filter_b = tf.Variable(tf.zeros(6))
conv1 = tf.nn.conv2d(output_h2, conv1_filter_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_filter_b
# Activation 1. Your choice of activation function.
conv1 = tf.nn.relu(conv1)
    # Pooling layer 1. The output shape should be 14x14x6.
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
   # Convolution layer 2. The output shape should be 10x10x16.
conv2_filter_w = tf.Variable(tf.truncated_normal([5,5,6,16]))
conv2_filter_b = tf.Variable(tf.zeros(16))
conv2 = tf.nn.conv2d(conv1, conv2_filter_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_filter_b
# Activation 2. Your choice of activation function.
conv2 = tf.nn.relu(conv2)
# Pooling layer 2. The output shape should be 5x5x16.
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
   # Flatten layer. Flatten the output shape of the final pooling layer
    # such that it's 1D instead of 3D. The easiest way to do is by using
    # tf.contrib.layers.flatten, which is already imported for you.
flatten_conv2 = flatten(conv2)
    # Fully connected layer 1. This should have 120 outputs.
    # 5 * 5 * 16 flatten is 400
fc1_input = flatten_conv2
fc1_w = tf.Variable(tf.truncated_normal([400, 120]))
fc1_b = tf.Variable(tf.zeros(120))
fc1_output = tf.matmul(fc1_input, fc1_w) + fc1_b
    # Activation 3. Your choice of activation function.
fc1_output = tf.nn.relu(fc1_output)
    # Fully connected layer 2. This should have 10 outputs.
fc2_w = tf.Variable(tf.truncated_normal([120, 10]))
fc2_b = tf.Variable(tf.zeros(10))
fc2_output = tf.matmul(fc1_output, fc2_w) + fc2_b

xentropy_n = tf.nn.softmax_cross_entropy_with_logits(logits=fc2_output,labels=Y)#prediction
loss_n = tf.reduce_mean(xentropy_n)


optimizer_n = tf.train.AdamOptimizer(learning_rate)
training_op_n = optimizer_n.minimize(loss_n,var_list=[conv1_filter_w,conv1_filter_b,conv2_filter_b,conv2_filter_w,fc1_w,fc1_b,fc2_w,fc2_b])


correct_f = tf.equal(tf.argmax(fc2_output,1),tf.argmax(Y,1))
accuracy_f = tf.reduce_mean(tf.cast(correct_f,tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	with tf.name_scope("phase1"):
		phase1_out = tf.matmul(output_h1,get_weights()[3])+get_biases()[3]
		reconstruction_loss_p1 = tf.reduce_mean(tf.square(phase1_out-X))
		reg_loss_p1 = l2_reg(get_weights()[3])
		
		loss_p1 = tf.add(reconstruction_loss_p1,reg_loss_p1)
		training_op_p1 = optimizer.minimize(loss_p1)


	with tf.name_scope("phase2"):
		reconstruction_loss_p2 = tf.reduce_mean(tf.square(output_h3-output_h1))
		reg_loss_p2 = l2_reg(get_weights()[1])+l2_reg(get_weights()[2])
		loss_p2 = tf.add(reconstruction_loss_p2,reg_loss_p2)
		train_vars = [get_weights()[2],get_weights()[1],get_biases()[2],get_biases()[1]]
		training_op_p2= optimizer.minimize(loss_p2,var_list=train_vars)



	for epoch in range(n_epochs):
		for i in range(n//batch_size):
			e = i + batch_size
			X_batch = X_train[i:e]
			Y_batch = Y_train[i:e]
			sess.run(training_op_p1,feed_dict={X:X_batch,Y:Y_batch})

	for epoch in range(n_epochs):
		for i in range(n//batch_size):
			e = i + batch_size
			X_batch = X_train[i:e]
			Y_batch = Y_train[i:e]
			sess.run(training_op_p2,feed_dict={X:X_batch,Y:Y_batch})



with tf.Session() as sess:  
	sess.run(tf.global_variables_initializer())
	for epoch in range(n_epochs):
		for i in range(n//batch_size):
			e = i + batch_size
			X_batch = X_train[i:e]
			Y_batch = Y_train[i:e]
			sess.run(training_op_n,feed_dict={X:X_batch,Y:Y_batch})		


	accu_val = accuracy_f.eval(feed_dict={X:mnist.validation.images,Y:mnist.validation.labels})
	print("accuracy :",accu_val)




