#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:38:53 2019

@author: slagghi
"""

# this file tests how the data from the dataset is structured
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
from PIL import Image
from cache import cache
import pickle
# from tf.keras.models import Model  # This does not work!
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import coco
import scipy.io as spy
coco.set_data_dir("../TensorFlow-Tutorials/data/coco/")
#coco.maybe_download_and_extract()
# THESE ARE THE COCO ORIGINAL FILENAMES AND CAPTIONS
#_, filenames_train, captions_train = coco.load_records(train=True)

# import the dataset filenames and captions
import json
path_json="../../dataset_rsicd.json"
with open(path_json) as f:
    DS=json.load(f)

#spy.loadmat("../../LSTM/captions.mat")

# read captions from text file
with open ("captions.txt", "r") as myfile:
    txt_captions=myfile.readlines()
# remove \n    
txt_captions=[x.strip() for x in txt_captions]
# save the captions in lists of five as a tuple
captions_train=()
filenames_train=()
captions_test=()
filenames_test=()
captions_val=()
filenames_val=()

for i in range(0,10921):
#    only consider train images for now
    split=DS['images'][i]['split']
    name=DS['images'][i]['filename']
    string_list=list()
    for j in range(0,5):
#        save the 5 captions for the image in a list
        string_list.append(txt_captions[5*i+j])
#    append the list to the corresponding tuple
    if split=='train':
        captions_train=captions_train+(string_list,)
        filenames_train=filenames_train+(name,)
    if split=='val':
        captions_val=captions_val+(string_list,)
        filenames_val=filenames_val+(name,)
    if split=='test':
        captions_test=captions_test+(string_list,)
        filenames_test=filenames_test+(name,)
    if i%500==0:
        print(25*'-')
        print('processed',i,'images')
        print('Training:\t',len(filenames_train))
        print('Test:\t',len(filenames_test))
        print('Val:\t',len(filenames_val))
        print(25*'-')

# save in json

with open('dataset/captions_train.json','w') as outfile:
    json.dump(captions_train,outfile)
with open('dataset/captions_test.json','w') as outfile:
    json.dump(captions_test,outfile)
with open('dataset/captions_val.json','w') as outfile:
    json.dump(captions_val,outfile)

with open('dataset/filenames_train.json','w') as outfile:
    json.dump(filenames_train,outfile)
with open('dataset/filenames_test.json','w') as outfile:
    json.dump(filenames_test,outfile)
with open('dataset/filenames_val.json','w') as outfile:
    json.dump(filenames_val,outfile)

# TO LOAD THE FILES
#with open(filename,'r') as infile:
#    data=json.load(infile)
