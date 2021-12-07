from os import sep
import pickle
import math
import random
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

UNK_TOKEN = "*UNK*"

def create_splits(lib, con, neutral, train_test_split=.8):
    '''

    Creates the total corpus, test data, and train data splits.

    :param lib: list containing liberal sentences
    :param con: list containung conservative sentences
    :param neutral: list containung neutral sentences
    :param train_test_split: percent of corpus to be used as training data
    :return: total corpus, train data, test data (in phrase format)
    '''
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
    '''
    Splits character phrases into list of words and labels.

    :param data: list of elements of shape (data_size, 2) where first row is phrase and second row is label
    :return phrase: list of phrases with each row in form of list of words
    :return labels: list of labels
    '''

    phrase = []
    labels = []

    for elt in data:
        phrase.append(elt[0].lower().split())
        labels.append(elt[1])

    return phrase, labels

def build_vocab(sentences):
    '''
    Builds dictionary of vocab with word as key and index as value.

    :param sentences: list of sentences
    :return vocab: dict of vocab
    '''
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set([UNK_TOKEN] + tokens)))
    vocab =  {word:i for i,word in enumerate(all_words)}
    return vocab

def convert_to_id(vocab, sentences):
    '''
    Converts sentences into corresponding ids from vocab.

    :param vocab: dict mapping word -> index
    :param sentneces: list of sentences
    :return ids: list of sentences in id (numerical) format
    '''
    ids = []
    for sentence in sentences:
        id = [vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence]
        ids.append(id)
    return ids

def batch(inputs, batch_size, batch_num, dataset_size):
    '''
    Batches input data according to batch_size.

    :param inputs: input data
    :param batch_size: batch size
    :param batch_num: batch number
    :param dataset_size: size of input data
    :return: batched inputs of size (batch_size, :)

    '''
    final_size = min(batch_size, dataset_size - (batch_num * batch_size))
    return inputs[batch_num * batch_size:batch_num * batch_size + final_size]

def get_data(train_data, test_data, train_labels, test_labels):
    '''
    Preprocesses the input data.

    :param input_file: pickled IBC file
    :return: train data IDs, test data IDs, train labels, tesst labels, and vocabulary
    '''

    # [lib, con, neutral] = pickle.load(open(input_file, 'rb'))

    # total_data, train_data, test_data = create_splits(lib, con, neutral)

    # train_data_phrases, train_data_labels = separate_labels(train_data)
    # test_data_phrases, test_data_labels = separate_labels(test_data)

    # open_file = open("train", "wb")
    # pickle.dump(train_data_phrases, open_file)
    # open_file.close()

    # open_file = open("test", "wb")
    # pickle.dump(test_data_phrases, open_file)
    # open_file.close()

    # open_file = open("train_labels", "wb")
    # pickle.dump(train_data_labels, open_file)
    # open_file.close()

    # open_file = open("test_labels", "wb")
    # pickle.dump(test_data_labels, open_file)
    # open_file.close()

    #open four files - train data, test data, train labels, test labels
    file_name = open(train_data, "rb")
    train_data_phrases = pickle.load(file_name)
    file_name.close()

    file_name = open(test_data, "rb")
    test_data_phrases = pickle.load(file_name)
    file_name.close()

    file_name = open(train_labels, "rb")
    train_data_labels = pickle.load(file_name)
    file_name.close()

    file_name = open(test_labels, "rb")
    test_data_labels = pickle.load(file_name)
    file_name.close()

    vocab = build_vocab(train_data_phrases)

    train_data_phrases = convert_to_id(vocab, train_data_phrases)
    test_data_phrases = convert_to_id(vocab, test_data_phrases)
    
    return train_data_phrases, test_data_phrases, train_data_labels, test_data_labels, vocab

