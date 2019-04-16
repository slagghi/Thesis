# This is a test for the capabilities
# of the python NLTK package
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from helpers import load_json
import numpy as np

# This helper function parses the tree for the desired pattern
def parse_tree(tree,label):
    retlist=list()
    for subtree in tree.subtrees():
        if subtree.label() == label:
            #print(label,": ",subtree.leaves())
            retlist.append(subtree)
    return retlist
# This function takes a relation or sub_obj tree and returns the string of its leaves
def get_words(tree):
    tree_len=len(tree)
    tree_words=list()
    for i in range(tree_len):
        tree_words.append(tree[i][0])
    return tree_words
# This function takes as input a string of words and concatenates them with spaces
def get_string(l):
    if len(l)==1:
        string=l[0]
    else:
        space=" "
        string=""
        for i in range(len(l)):
            string=string+l[i]+space
        string=string[:-1]
    return string

# This function gets the relation list starting from the text
def get_relation_list(text):
    tree=get_tree(text)
# This is a list of all the relations found in the sentence
    tree_list=parse_tree(tree,'RELATION')
# For each relation found in the sentence, save the subject, object and relation predicate
    relation_list=list()
    for i in range(len(tree_list)):
        relation_predicate=parse_tree(tree_list[i],'RELATION_PREDICATE')
        subject_object=parse_tree(tree_list[i],'SUBJECT_OBJECT')
#    get the string for the relationship predicate and subject_object
        relation_predicate_words=get_words(relation_predicate[0])
        subject_tree=subject_object[0]
        object_tree=subject_object[1]
        subject_words=get_words(subject_tree)
        object_words=get_words(object_tree)
        subject_string=get_string(subject_words)
        object_string=get_string(object_words)
        predicate_string=get_string(relation_predicate_words)
#    Save the relation in a triplet
        triplet=list()
        triplet.append(subject_string)
        triplet.append(predicate_string)
        triplet.append(object_string)
        relation_list.append(triplet)    
    return relation_list

# get the caption tree
def get_tree(text):
    grammar = """SUBJECT_OBJECT: {<NN|VBG>*<NN|NNS>}
    NOUN: {<DT|CD>?<DT|CD>?<JJ>*<SUBJECT_OBJECT>}
    RELATION_PREDICATE: {<VBN>?<IN>}
                        {<JJ><TO>}
                        {<VBN>?<RB>}
    VERB_PHRASE: {<VBP|VBZ|VBD|VBN|VBG>?<VBP|VBZ|VBD|VBN|VBG>}
    RELATION: {<NOUN><VERB_PHRASE>?<RELATION_PREDICATE><NOUN>}"""
    words=word_tokenize(text)
    pos_tags=nltk.pos_tag(words)
    chunkParser=nltk.RegexpParser(grammar)
    tree=chunkParser.parse(pos_tags)
    return tree


# load the train captions
captions_train=load_json('captions_train')
relation_list=list()
item_list=list()
predicate_list=list()
for i in range(8734):
    image_relation_list=list()
    image_item_list=list()
    image_predicate_list=list()
    for j in range(5):
        text=captions_train[i][j]
        r=get_relation_list(text)
        if r not in image_relation_list:
            if len(r)!=0:
                image_relation_list.append(r)
                for pred in r:
                    image_predicate_list.append(pred[1])
                items=parse_tree(get_tree(text),'SUBJECT_OBJECT')
                for item in items:
                    item_words=get_words(item)
                    item_string=get_string(item_words)
                    if item_string in image_item_list:
                        continue
                    image_item_list.append(item_string)
    if i%100==0:
        print(100*i/8734,"% complete")
    item_list.append(image_item_list)
    relation_list.append(image_relation_list)
    predicate_list.append(image_predicate_list)