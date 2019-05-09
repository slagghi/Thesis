# This file contains conde for the bag of words analysis
from helpers import load_json
import nltk
import re
from nltk.corpus import stopwords
import numpy as np
from helpers import print_progress
set(stopwords.words('english'))

def get_caption_list():
    captions_train=load_json('captions_train')
    fullCaptionList=list()
    for captions in captions_train:
        for caption in captions:
            fullCaptionList.append(caption)
    return fullCaptionList
            

def word_extraction(sentence):
    words=re.sub("[^\w]"," ",sentence).split()
    cleaned_text=[w.lower() for w in words]
    return cleaned_text

def tokenize(sentences):
    words=[]
    for sentence in sentences:
        w=word_extraction(sentence)
        words.extend(w)
        
    words=sorted(list(set(words)))
    return words

def generate_bow(allsentences,verbose=0):
    vocab=tokenize(allsentences)
    if verbose:
        print("Word List for Document \n{0} \n".format(vocab));
        
    ctr=0
    bow_total=np.zeros(len(vocab))
    image_bags=list()
    
    for sentence in allsentences:
        words=word_extraction(sentence)
        bag_vector=np.zeros(len(vocab))
        for w in words:
            for i,word in enumerate(vocab):
                if word==w:
                    bag_vector[i]+=1
        if verbose:
            print("{0}\n{1}\n".format(sentence,np.array(bag_vector)))
        bow_total=bow_total+bag_vector
        image_bags.append(bag_vector)
        ctr+=1
        print_progress(ctr,len(allsentences))
    return bow_total,vocab,image_bags

