import pickle
import math
import random


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

def create_splits(lib, con, neutral, train_test_split=.9):
    corpus = []
    labels = {"Liberal":0, "Conservative":1, "Neutral":2}
    for tree in lib:
       phrase_data = []
       for node in tree:
           if hasattr(node, 'label'):
               phrase_data.append([node.get_words(),labels[node.label]])
       #print(phrase_data)
       #corpus.append({"sentence": tree.get_words(), "label": 0, "phrases":phrase_data })
       corpus += phrase_data
    count = 0
    for tree in con:
        phrase_data = []
        for node in tree:
           if hasattr(node, 'label'):
               phrase_data.append([node.get_words(),labels[node.label]])
      #corpus.append({"sentence": tree.get_words(), "label": 1, "phrases":phrase_data })
        if count == 0:
            print(phrase_data)
            count += 1
        corpus += phrase_data
    for tree in neutral:
        phrase_data = []
        for node in tree:
           if hasattr(node, 'label'):
               phrase_data.append([node.get_words(),labels[node.label]])
        corpus += phrase_data
        #corpus.append({"sentence": tree.get_words(), "label": 2, "phrases":phrase_data })

    print(len(corpus))
    print(corpus[:3])
    

    cutoff = int(math.floor(train_test_split * len(corpus)))

    print('cuttoff: ', cutoff)

    random.shuffle(corpus)
    train_data = corpus[:cutoff]
    test_data = corpus[cutoff:]
    return corpus, train_data, test_data


if __name__ == '__main__':
    [lib, con, neutral] = pickle.load(open('ibcData.pkl', 'rb'))

    # how to access sentence text
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


    corpus, lib_sentences, con_sentences, neutral_sentences = extract_corpus(lib, con, neutral)
    total_data, train_data, test_data = create_splits(lib, con, neutral)
