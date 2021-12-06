from os import sep
import pickle
import math
import random
import numpy as np

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
#WINDOW_SIZE = 12

def extract_corpus(lib, con, neutral):
    corpus = []
    lib_sentences = []
    con_sentences = []
    neutral_sentences = []

    for tree in lib:
        corpus.append(tree.get_words())
        lib_sentences.append(tree.get_words())
    for tree in con:
        corpus.append(tree.get_words())
        con_sentences.append(tree.get_words())
    for tree in neutral:
        corpus.append(tree.get_words())
        neutral_sentences.append(tree.get_words())

    return corpus, lib_sentences, con_sentences, neutral_sentences

def create_splits(lib, con, neutral, train_test_split=.8):
    corpus = []
    labels = {"Liberal":0, "Conservative":1, "Neutral":2}
    for tree in lib:
       phrase_data = []
       for node in tree:
           if hasattr(node, 'label'):
               phrase_data.append([node.get_words(),labels[node.label]])
       corpus += phrase_data
    count = 0
    for tree in con:
        phrase_data = []
        for node in tree:
           if hasattr(node, 'label'):
               phrase_data.append([node.get_words(),labels[node.label]])
        corpus += phrase_data
    for tree in neutral:
        phrase_data = []
        for node in tree:
           if hasattr(node, 'label'):
               phrase_data.append([node.get_words(),labels[node.label]])
        corpus += phrase_data    

    cutoff = int(math.floor(train_test_split * len(corpus)))

    print('cuttoff: ', cutoff)

    random.shuffle(corpus)
    train_data = corpus[:cutoff]
    test_data = corpus[cutoff:]
    return corpus, train_data, test_data

def separate_labels(data):
    phrase = []
    labels = []
    max_len = 0
    for elt in data:
        if len(elt[0].split()) > max_len:
            max_len = len(elt[0].lower().split())
        phrase.append(elt[0].lower().split())
        labels.append(elt[1])

    print(max_len)

    return phrase, labels, max_len

def pad_corpus(sentences, window_size):
    print("Window size: " + str(window_size))
    padded_sentences = []
    for line in sentences:
        padded = line[:window_size]
        padded += [STOP_TOKEN]
        while len(padded) < window_size + 1:
            padded += [PAD_TOKEN]
        padded_sentences.append(padded)
    return padded_sentences

def build_vocab(sentences):
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))
	vocab =  {word:i for i,word in enumerate(all_words)}
	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
    ids = []
    for sentence in sentences:
        id = [vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence]
        ids.append(id)
    #return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])
    return ids

def batch(inputs, batch_size, batch_num, dataset_size):
    final_size = min(batch_size, dataset_size - (batch_num * batch_size))
    return inputs[batch_num * batch_size:batch_num * batch_size + final_size]

def get_data(input_file):
    [lib, con, neutral] = pickle.load(open(input_file, 'rb'))

    # how to access sentence text

    '''

    print('Liberal examples (out of ', len(lib), ' sentences): ')
    for tree in lib[0:5]:
        print(tree.get_words())

    print('\nConservative examples (out of ', len(con), ' sentences): ')
    for tree in con[0:5]:
        print(tree.get_words())

    print('\nNeutral examples (out of ', len(neutral), ' sentences): ')
    for tree in neutral[0:5]:
        print(tree.get_words())

    

    # how to access phrase labels for a particular tree
    ex_tree = lib[0]

    print('\nPhrase labels for one tree: ')

    # see treeUtil.py for the tree class definition
    for node in ex_tree:

        # remember, only certain nodes have labels (see paper for details)
        if hasattr(node, 'label'):
            print(node.label, ': ', node.get_words())

    '''

    #corpus, lib_sentences, con_sentences, neutral_sentences = extract_corpus(lib, con, neutral)
    total_data, train_data, test_data = create_splits(lib, con, neutral)

    train_data_phrases, train_data_labels, train_max = separate_labels(train_data)
    test_data_phrases, test_data_labels, test_max = separate_labels(test_data)

    window_size = max(train_max, test_max)

    # train_data_phrases = pad_corpus(train_data_phrases, window_size)
    # test_data_phrases = pad_corpus(train_data_phrases, window_size)
    
    vocab, padding_index = build_vocab(train_data_phrases)
    train_data_phrases = convert_to_id(vocab, train_data_phrases)
    test_data_phrases = convert_to_id(vocab, test_data_phrases)

    #print(np.shape(train_data_phrases))
    
    return train_data_phrases, test_data_phrases, train_data_labels, test_data_labels, vocab, window_size
