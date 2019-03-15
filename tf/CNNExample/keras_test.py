#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:32:05 2019

@author: slagghi
"""

# this code is to get familiarity with keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import h5py

#i mport keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils.vis_utils import plot_model

# tensorboard visualization
tbCallBack =TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Loading the dataset
from mnist import MNIST
data=MNIST(data_dir="data/MNIST/")

img_size=data.img_size
img_size_flat=data.img_size_flat
img_shape=data.img_shape
img_shape_full=data.img_shape_full
num_classes=data.num_classes
num_channels=data.num_channels

# Helper functions
def plot_images(images,cls_true,cls_pred=None):
    assert len(images)==len(cls_true)==9
    
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i,ax in enumerate(axes.flat):
        # plot image
        ax.imshow(images[i].reshape(img_shape),cmap='binary')
        if cls_pred is None:
            xlabel="True: {0}".format(cls_true[i])
        else:
            xlabel="True: {0}, Pred: {1}".format(cls_true[i],cls_pred[i])
        # Classes are the x label
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
def plot_example_errors(cls_pred):
    # this array verifies if the class is incorrect
    incorrect=(cls_pred!=data.y_test_cls)
    
    # get the wrongly classified images in the testset
    images=data.x_test[incorrect]
    cls_pred=cls_pred[incorrect]
    cls_true=data.y_test_cls[incorrect]
    
    # plot the first 9 incorrect images
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# MODEL DEFINITION
# using KERAS sequential mode

model=Sequential()
# input layer
# input shape is a tuple containing image size
model.add(InputLayer(input_shape=(img_size_flat,)))
# the input is a 784 element flattened array
# so it must be reshaped in an image
model.add(Reshape(img_shape_full))
# conv1 with ReLU-acivation and max-pooling
model.add(Conv2D(kernel_size=5,strides=1,filters=16,padding='same',
                 activation='relu',name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2,strides=2))
# conv2
model.add(Conv2D(kernel_size=5,strides=1,filters=36,padding='same',
                 activation='relu',name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2,strides=2))
# flatten the convolutional output into one-dimensional
# so it can be fed to a fully-connected layer
model.add(Flatten())
model.add(Dense(128,activation='relu'))
# softmax
model.add(Dense(num_classes,activation='softmax'))


# Loss function and optimizer
from tensorflow.python.keras.optimizers import Adam
optimizer=Adam(lr=1e-3)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model.fit(x=data.x_train,
#              y=data.y_train,
#              epochs=1,
#              batch_size=128)


# training
def train_network(n_epochs=1):
    model.fit(x=data.x_train,
              y=data.y_train,
              epochs=n_epochs,
              batch_size=128,
              callbacks=[tbCallBack])
def evaluate_network():
    result=model.evaluate(x=data.x_test,
                          y=data.y_test)
    for name,value in zip(model.metrics_names,result):
        print(name,value)
def test_network(start_i):
    # this function tests the image on nine images
    # and shows the predicted class
    end_i=start_i+9
    images=data.x_test[start_i:end_i]
    cls_true=data.y_test_cls[start_i:end_i]
    # confidence array for each class
    y_pred=model.predict(x=images)
    # choose the most confident
    cls_pred=np.argmax(y_pred,axis=1)
    plot_images(images=images,
                cls_true=cls_true,
                cls_pred=cls_pred)

def display_errors():
    y_pred=model.predict(x=data.x_test)
    cls_pred=np.argmax(y_pred,axis=1)
    plot_example_errors(cls_pred)

#path_model='model.keras'
#model.save(path_model)
#model2=load_model(path_model)


    