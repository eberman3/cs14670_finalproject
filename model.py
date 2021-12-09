import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding
from loadIBC import UNK_TOKEN
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


class MLP(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(MLP, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = 512 
        self.batch_size = 100 

        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1))

        self.model = Sequential(name='MLP')
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=256, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=3, activation='softmax'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self, inputs):
        """
        Calls the model on a batch of data.

        :param inputs: batch of data - each row is a sentence
        :return: None
        """

        sentence_matrix = []
        for row in inputs:
            embedded_row = tf.nn.embedding_lookup(self.E, row)
            embedded_row = tf.reduce_mean(embedded_row, axis=0)
            sentence_matrix.append(embedded_row)
        sentence_matrix = tf.Variable(sentence_matrix)
        return self.model(sentence_matrix)

    def loss(self, probs, labels):
        """
        Calculates average cross entropy loss
        
        :param probs: a matrix of shape (batch_size, 3) containing probabilities for each class
        :param labels: matrix of shape (batch_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))

    def plot_similarities(self, vocab, sentences):
        print("Now plotting...")

        ids = []
        for sentence in sentences:
            split = sentence[0].lower().split()
            id = [vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in split]
            ids.append(id)

        vectors = []
        for row in ids:
            vectors.append(tf.reduce_mean(tf.nn.embedding_lookup(self.E, row).numpy(), axis=0))
        
        pca = PCA(n_components=2)
        vectors = pca.fit_transform(vectors)

        normalizer = preprocessing.Normalizer()
        norm_vectors = normalizer.fit_transform(vectors, 'l2')

        # plot the 2D normalized vectors
        x_vec = []
        y_vec = []
        for x,y in norm_vectors:
            x_vec.append(x)
            y_vec.append(y)
        
        f, axs = plt.subplots(1,1,figsize=(7,4))
        plt.scatter(x_vec, y_vec)
        i = 0
        for sentence in sentences:
            plt.annotate(sentence[0], (norm_vectors[i][0], norm_vectors[i][1]))
            i += 1
        plt.show()

