# this code evaluates the captions generated using beamsearch
import json
import nltk
import math
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from helpers import load_json
from helpers import print_progress
from helpers import load_image
import numpy as np
img_size=(228,228)


from Scores import consensus_score
chencherry=SmoothingFunction()


# LOAD THE TRANSFER MODEL
# can do thi only at first execution
#
#from tensorflow.python.keras.applications import VGG16
#from tensorflow.python.keras.models import Model
#image_model = VGG16(include_top=True, weights='imagenet')
#transfer_layer=image_model.get_layer('fc2')
#image_model_transfer = Model(inputs=image_model.input,
#                             outputs=transfer_layer.output)
def get_transfer_values(image_path):
    '''
    Compute the transfer values for an image
    given the image transfer model
    '''
    
    # Load and resize the image.
    image = load_image(image_path, size=img_size)
    
    # Expand the 3-dim numpy array to 4-dim
    # because the image-model expects a whole batch as input,
    # so we give it a batch with just one image.
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = image_model_transfer.predict(image_batch)
    return transfer_values



filename='test_captions_beamsearch_VGG16.json'
with open(filename,'r') as inFile:
    beamCaptions=json.load(inFile)
    beamCaptions=tuple(beamCaptions)

captions_test=load_json('captions_test')
filenames_test=load_json('filenames_test')
transfer_values_train=np.load('dataset/transfer_values_train.npy')
captions_train=load_json('captions_train')

# build the references library
num_samples=1093
references=list()
for i in range(num_samples):
    R=captions_test[i]
    refList=list()
    for j in range(5):
        refList.append(nltk.word_tokenize(R[j]))
    references.append(refList)
    
print('Choosing best caption based on blue score (cheating)')
bestCaptions=list()
idxList=list()
for i in range(num_samples):
    bleuSums=list()
    for j in range(5):
        candidate=beamCaptions[i][j]['sentence']
        candidate_tokenized=nltk.word_tokenize(candidate)

        s1=sentence_bleu(references[i],candidate_tokenized,weights=[1,0,0,0],smoothing_function=chencherry.method1)
        s2=sentence_bleu(references[i],candidate_tokenized,weights=[0.5,0.5,0,0],smoothing_function=chencherry.method1)
        s3=sentence_bleu(references[i],candidate_tokenized,weights=[0.33,0.33,0.33,0],smoothing_function=chencherry.method1)
        s4=sentence_bleu(references[i],candidate_tokenized,weights=[0.25,0.25,0.25,0.25],smoothing_function=chencherry.method1)

# First approach: the best sentence is the one with the best
# BLEU sum
        bleuSum=s1+s2+s3+s4
        bleuSums.append(bleuSum)
    bestIdx=np.argmax(bleuSums)
#    print(bestCaption,bleuSums)
    bestCaption=beamCaptions[i][bestIdx]['sentence']
    bestCaptions.append(bestCaption[1:])
    idxList.append(bestIdx)
#    plot progress
    print_progress(i,num_samples)
    
# aggregate the corpus of the best caption for each image
#bestCorpus=list()
#for i in range(num_samples):
#    idx=bestCaptions[i]
#    candidate=beamCaptions[i][idx]['sentence']
#    bestCorpus.append(candidate)
    

# SECON METHOD
# the best caption is the one with the best consensus score
    

    
print('\nChoosing best caption based on consensus score')
image_dir='../../../../Desktop/parsingDataset/RSICD_images/'
idxList2=list()
bestCaptions2=list()

for i in range(num_samples):
    consScores=list()
    image_filename=filenames_test[i]
    if i==884:
        image_filename=filenames_test[883]
    transfer_values=get_transfer_values(image_dir+image_filename)
    
    for j in range(5):
        candidate=beamCaptions[i][j]['sentence']

        score=consensus_score(candidate,transfer_values)
        consScores.append(score)
    bestIdx=np.argmax(consScores)
    idxList2.append(bestIdx)    
    bestCaption=beamCaptions[i][bestIdx]['sentence']
    bestCaptions2.append(bestCaption[1:])

    print_progress(i,num_samples)
    
    