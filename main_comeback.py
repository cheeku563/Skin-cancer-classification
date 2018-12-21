from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf 

import cv2

from tensorflow.contrib.layers import flatten
import numpy as np 

import matplotlib.pyplot as plt
from math import sqrt

import PIL
import os
import numpy as np
import pandas as pd  
from PIL import Image

X_train = []
n = 10015
Y_train = np.zeros((n,7))

data_frame = pd.read_csv(r'C:\Users\vsaik\Anaconda3\envs\tensorflow\TensorFlow-Tutorials\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv',sep=',')


for i in range(n):
	for j in range(1,8):
		Y_train[i][j-1] = data_frame.loc[i][j]


for number,key in enumerate(os.listdir(r'C:\Users\vsaik\Anaconda3\envs\tensorflow\TensorFlow-Tutorials\ISIC2018_Task3_Training_Input')):
	if(number<=n):
		img= cv2.imread(os.path.join(r'C:\Users\vsaik\Anaconda3\envs\tensorflow\TensorFlow-Tutorials\ISIC2018_Task3_Training_Input',key))
		x = img.flatten()
		X_train.append(x)

X_validation=[]

for number,key in enumerate(os.listdir(r'C:\Users\vsaik\Anaconda3\envs\tensorflow\TensorFlow-Tutorials\ISIC2018_Task3_Validation_Input')):
	if(number<=n):
		img= cv2.imread(os.path.join(r'C:\Users\vsaik\Anaconda3\envs\tensorflow\TensorFlow-Tutorials\ISIC2018_Task3_Validation_Input',key))
		x = img.flatten()
		X_validation.append(x)
		
		
		


from functools import partial


n_epochs = 3
batch_size=50
learning_rate=0.1

n_inputs = 600*450*3
n_hidden_1 = 2000
n_hidden_2 = 1024
n_hidden_3 = 512
n_hidden_4 = 256
n_hidden_5 = 128
n_hidden_6 = 64
n_hidden_7 = 128
n_hidden_8 = 256
n_hidden_9 = 512
n_hidden_10 = 1024
n_hidden_11 = 2000
n_outputs=600*450*3

l2 = 0.001

X = tf.placeholder(tf.float32,shape=(None,600*450*3))
Y = tf.placeholder(tf.float32,shape = (None,7))

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
output_h4 = tf.layers.dense(output_h3,n_hidden_4,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h5 = tf.layers.dense(output_h4,n_hidden_5,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h6 = tf.layers.dense(output_h5,n_hidden_6,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h7 = tf.layers.dense(output_h6,n_hidden_7,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h8 = tf.layers.dense(output_h7,n_hidden_8,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h9 = tf.layers.dense(output_h8,n_hidden_9,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h10 = tf.layers.dense(output_h9,n_hidden_10,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)
output_h11 = tf.layers.dense(output_h10,n_hidden_11,activation = acti,kernel_initializer= he_init,kernel_regularizer = l2_reg)





output = tf.layers.dense(output_h11,n_inputs,activation = None,kernel_initializer= he_init,kernel_regularizer = l2_reg)

reconstruction_loss = tf.reduce_mean(tf.square(output-X))
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add(reconstruction_loss,reg_loss)


optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)



nn = tf.layers.dense(output_h6,100,activation=tf.nn.relu,kernel_initializer=he_init,kernel_regularizer=l2_reg)
#nn2 = tf.layers.dense(nn,50,activation=tf.nn.relu,kernel_initializer=he_init,kernel_regularizer=l2_reg)
logits = tf.layers.dense(nn,7,activation=None,kernel_initializer=he_init,kernel_regularizer=l2_reg)

output_f = tf.sigmoid(logits)

xentropy_n = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)#prediction
loss_n = tf.reduce_mean(xentropy_n)

optimizer_n = tf.train.AdamOptimizer(learning_rate)
training_op_n = optimizer_n.minimize(loss_n,var_list=[get_weights()[12],get_weights()[13],get_biases()[12],get_biases()[13]])



correct_f = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
accuracy_f = tf.reduce_mean(tf.cast(correct_f,tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess,r"C:\Users\vsaik\Anaconda3\envs\tensorflow\TensorFlow-Tutorials\ae_nn_values_saiky_2.ckpt")
	print("Model restored.")
	X_batch = X_validation
	encodings = sess.run(output_h6,feed_dict={X:X_batch})
	out = sess.run(output_f,feed_dict={X:X_batch})
	new = tf.argmax(out,1)
	print(new)



