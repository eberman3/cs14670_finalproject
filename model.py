import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from loadIBC import batch


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