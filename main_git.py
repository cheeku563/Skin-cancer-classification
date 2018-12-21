from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf 

import cv2

from tensorflow.contrib.layers import flatten
import numpy as np 

from sklearn.metrics import classification_report
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

data_frame = pd.read_csv(r'/lustre/work/deogun/vsaiky563/TensorFlow-Tutorials/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv',sep=',')


for i in range(n):
	for j in range(1,8):
		Y_train[i][j-1] = data_frame.loc[i][j]


for number,key in enumerate(os.listdir(r'/lustre/work/deogun/vsaiky563/TensorFlow-Tutorials/ISIC2018_Task3_Training_Input')):
	if(number<=n):
		img= cv2.imread(os.path.join(r'/lustre/work/deogun/vsaiky563/TensorFlow-Tutorials/ISIC2018_Task3_Training_Input',key))
		x = img.flatten()
		X_train.append(x)
		
		


from functools import partial

X_validation = X_train[8:10]
Y_validation = Y_train[8:10]

n_epochs = 4
n_epochs_2 = 3
batch_size=5
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


def plot_image(image,shape):
	plt.imshow(image.reshape(shape),cmap="Greys",interpolation = "nearest")
	plt.axis("off")



acti = tf.nn.sigmoid
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



nn = tf.layers.dense(output_h6,60,activation=tf.nn.relu,kernel_initializer=he_init,kernel_regularizer=l2_reg)
nn2 = tf.layers.dense(nn,50,activation=tf.nn.relu,kernel_initializer=he_init,kernel_regularizer=l2_reg)
logits = tf.layers.dense(nn2,7,activation=None,kernel_initializer=he_init,kernel_regularizer=l2_reg)

output_f = tf.sigmoid(logits)

xentropy_n = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)#prediction
loss_n = tf.reduce_mean(xentropy_n)

optimizer_n = tf.train.AdamOptimizer(learning_rate)
training_op_n = optimizer_n.minimize(loss_n,var_list=[get_weights()[12],get_weights()[13],get_biases()[12],get_biases()[13],get_biases()[14],get_weights()[14]])



correct_f = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
accuracy_f = tf.reduce_mean(tf.cast(correct_f,tf.float32))

saver = tf.train.Saver()

n = 10015

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(n_epochs):
		for i in range(n//batch_size):
			e = i + batch_size
			X_batch = X_train[i:e]
			sess.run(training_op,feed_dict={X:X_batch})
		a = sess.run(output_h6,feed_dict={X:X_batch})
		plot_image(output_h6,shape=[8,8,1])
		plt.show()
		plot_image(output,shape=[450,600,3])
		plt.show()	
		print("output:",a)

	for epoch in range(n_epochs_2):
		for i in range(n//batch_size):
			e = i + batch_size
			X_batch = X_train[i:e]
			Y_batch = Y_train[i:e]
			sess.run(training_op_n,feed_dict={X:X_batch,Y:Y_batch})		
		b = sess.run(output_f,feed_dict={X:X_batch})
		print("nn_out :",b)

	accu_val = accuracy_f.eval(feed_dict={X:X_validation,Y:Y_validation})
	print("accuracy :",accu_val)
	
	Y_test = np.argmax(Y_validation, axis=1) 
	y_p = sess.run(output_f,feed_dict={X:X_validation})
	y_pred = np.argmax(y_p,axis=1)

	print(classification_report(Y_test, y_pred))

	confusion_m = tf.confusion_matrix(labels=Y_test,predictions=y_pred,num_classes=7,dtype=tf.int32,weights=None)
	print(sess.run(confusion_m))'''

	print("done")




