from helpers import load_image
import numpy as np

# This code implements beam search for a less-greedy sentence generation

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

# ASSUME I ALREADY HAVE THE TRANSFER VALUES FOR THE IMAGE
filenames_test=load_json('filenames_test')
path='../../../../Desktop/parsingDataset/RSICD_images/'
filename=filenames_test[812]
image = load_image(image_path, size=img_size)
image_batch = np.expand_dims(image, axis=0)
transfer_values = image_model_transfer.predict(image_batch)

# This code, given a transfer vector and the previous sequence, predicts the
# K best next words (start with 2)
prev_sequence=[]

def predict_next_word(transfer_values,prev_sequence):
    count_tokens=len(prev_sequence)
    shape=(1,30)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    for i in range(count_tokens):
        decoder_input_data[0,i]=prev_sequence[i]
    x_data={'transfer_values_input': transfer_values,'decoder_input': decoder_input_data}
    decoder_output = decoder_model.predict(x_data)
#    NB: non-normalised tokens are negative
#    DON'T NORMALISE TOKENS
#    otherwise 1.0 is always gonna win, as it's always the RELATIVE best
#   while we are interested in the absolute best
    token_onehot = decoder_output[0, count_tokens, :]
    first_choice=[np.argmax(token_onehot),np.max(token_onehot)]
    return first_choice

# This code normalises a vector in a [0,1] range
def normalise(vector):
    M=np.max(vector)
    m=np.min(vector)
    vector=(vector-m)/(M-m)
    return vector