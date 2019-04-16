from helpers import load_image
import numpy as np
from helpers import load_json

from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

from captions_preprocess import TokenizerWrap
from captions_preprocess import flatten
from captions_preprocess import mark_captions

# This code implements beam search for a less-greedy sentence generation
# NOTE:
#   Before running this code, the NN_architecture code should be run beforehand
#   since it contains the model for the decoder

def generate_caption(image_path, max_tokens=30):
    """
    Generate a caption for the image in the given path.
    The caption is limited to the given number of tokens (words).
    """

    # Load and resize the image.
    image = load_image(image_path, size=img_size)
    
    # Expand the 3-dim numpy array to 4-dim
    # because the image-model expects a whole batch as input,
    # so we give it a batch with just one image.
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = image_model_transfer.predict(image_batch)

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # This is the sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Plot the image.
#    plt.imshow(image)
#    plt.title(output_text.replace(" eeee",""))
#    plt.axis('off')
#    plt.show()
#    plt.savefig("test_results/test.png", bbox_inches='tight')
    
    # Print the predicted caption.
#    print("Predicted caption:")
#    print(output_text.replace(" eeee",""))
#    print()
    return output_text.replace(" eeee","")

img_size=(228,228)

# Load the transfer model (CNN)
image_model = VGG16(include_top=True, weights='imagenet')
transfer_layer=image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
img_size=K.int_shape(image_model.input)[1:3]

# recreate the tokenizer
mark_start='ssss '
mark_end=' eeee'
captions_train=load_json('captions_train')
captions_train_marked=mark_captions(captions_train)
captions_train_flat=flatten(captions_train_marked)
tokenizer=TokenizerWrap(texts=captions_train_flat,
                        num_words=2000)

token_start=tokenizer.word_index[mark_start.strip()]
token_end=tokenizer.word_index[mark_end.strip()]

# ASSUME I ALREADY HAVE THE TRANSFER VALUES FOR THE IMAGE
filenames_test=load_json('filenames_test')
path='../../../../Desktop/parsingDataset/RSICD_images/'
filename=filenames_test[812]
image_path=path+filename
image = load_image(image_path, size=img_size)
image_batch = np.expand_dims(image, axis=0)
transfer_values = image_model_transfer.predict(image_batch)




# This code, given a transfer vector and the previous sequence, predicts the
# K best next words (start with 2)
prev_sequence=np.zeros(shape=(1,30),dtype=np.int)
prev_sequence[0,0]=token_start
count_tokens=0

def predict_next_word(transfer_values,prev_sequence,count_tokens,guessNr):
    x_data=\
    {
     'transfer_values_input':transfer_values,
     'decoder_input':prev_sequence
     }
    decoder_output=decoder_model.predict(x_data)
    token_onehot = decoder_output[0, count_tokens, :]

    [outToken,confidence]=nth_best(token_onehot,guessNr)
    outWord=tokenizer.token_to_word(outToken)
    
    return outToken,confidence

def beam_search(transfer_value):
#    Cycle on words, with a max of 30
#   the first predicted sequence is just the start token
    prev_sequence=np.zeros(shape=(1,30),dtype=np.int)
    prev_sequence[0,0]=token_start
    count_tokens=1
    nr_guesses=3
    caption_list=list()
    starter_caption={
            'sequence':prev_sequence,
            'confidence':0
            }
    caption_list.append(starter_caption)
    for count_tokens in range(1,2):
#        for each caption in the list compute nr_guesses new captions and put them in a list
        new_captions=list()
        for caption in caption_list:
            prev_sequence=caption['sequence']
            prev_confidence=caption['confidence']
            for guess_iter in range(1,nr_guesses):
                [token,confidence]=predict_next_word(transfer_values,prev_sequence,count_tokens,guess_iter)
#                update the sequence
                new_sequence=copy.copy(prev_sequence)
                new_sequence[0,count_tokens]=token
                new_caption={
                        'sequence':copy.copy(new_sequence),
                        'confidence':prev_confidence+confidence
                        }
                new_captions.append(copy.copy(new_caption))
        caption_list=list()
        caption_list=new_captions[:]
    return caption_list
# This function returns the n-th highest value in a vector
def nth_best(vector,n):
    v=vector
#    discard (n-1) biggest elements
    for i in range(n-1):
        best=np.argmax(v)
        v[best]=-100
    best_position=np.argmax(v)
    best_value=max(v)
    return best_position,best_value
# This code normalises a vector in a [0,1] range
def normalise(vector):
    M=np.max(vector)
    m=np.min(vector)
    vector=(vector-m)/(M-m)
    return vector