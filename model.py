import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from loadIBC import batch


class MLP(tf.keras.Model):
    def __init__(self, vocab_size, window_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(MLP, self).__init__()

        # TODO: initialize vocab_size, embedding_size

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_size = 512 #TODO
        self.batch_size = 100 #TODO 

        self.E = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1))

        self.model = Sequential(name='model')

        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=256, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=3, activation='softmax'))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """

        #TODO: Fill in 
        #print(inputs[0:5])
        sentence_matrix = []
        for row in inputs:
            embedded_row = tf.nn.embedding_lookup(self.E, row)
            #print(embedded_row)
            embedded_row = tf.reduce_mean(embedded_row, axis=0)
            sentence_matrix.append(embedded_row)
        sentence_matrix = tf.Variable(sentence_matrix)

        return self.model(sentence_matrix)

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(labels, probs))


def train(model, train_inputs, train_labels, test_inputs, test_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #TODO: Fill in
    num_batches = len(train_inputs) // model.batch_size
    losses = []

    inputs = train_inputs
    labels = train_labels
    for num_epoch in range(25):
        print("Num epoch: " + str(num_epoch))
        total_loss = 0
        for batch_num in range(num_batches):
            curr_batch = batch(inputs, model.batch_size, batch_num, len(inputs))
            curr_batch_labels = batch(labels, model.batch_size, batch_num, len(inputs))

            with tf.GradientTape() as tape:
                probs = model.call(curr_batch)
                loss = model.loss(probs, curr_batch_labels)
                total_loss += loss
            
            if (batch_num % 50 == 0):
                print("Batch num " + str(batch_num) + " validation: " + str(test(model, test_inputs, test_labels)))

            gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        total_loss / num_batches
        losses.append(total_loss)
        
        # indices = np.arange(len(inputs))
        # indices = tf.random.shuffle(indices)

        # inputs = tf.gather(inputs, indices)
        # labels = tf.gather(labels, indices)
    
    visualize_loss(losses)


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """

    num_batches = len(test_inputs) // model.batch_size

    #print(num_batches)

    #total_loss = 0
    total_accuracy = 0

    for batch_num in range(num_batches):

        curr_batch = batch(test_inputs, model.batch_size, batch_num, len(test_inputs))
        curr_batch_labels = batch(test_labels, model.batch_size, batch_num, len(test_inputs))

        probs = model.call(curr_batch)
        predictions = np.argmax(probs, axis=1)
        total_accuracy += np.mean(predictions == curr_batch_labels)

        #total_loss += model.loss(probs, curr_batch_labels)

    return total_accuracy / num_batches

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. Call this in train().
    When you observe the plot that's displayed, think about:
    1. What does the plot demonstrate or show?
    2. How long does your model need to train to reach roughly its best accuracy so far, 
    and how do you know that?
    Optionally, add your answers to README!
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT
    
    :return: doesn't return anything, a plot should pop-up
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()


# def generate_sentence(word1, length, vocab, model, sample_n=10):
#     """
#     Takes a model, vocab, selects from the most likely next word from the model's distribution

#     :param model: trained RNN model
#     :param vocab: dictionary, word to id mapping
#     :return: None
#     """

#     #NOTE: Feel free to play around with different sample_n values

#     reverse_vocab = {idx: word for word, idx in vocab.items()}
#     previous_state = None

#     first_string = word1
#     first_word_index = vocab[word1]
#     next_input = [[first_word_index]]
#     text = [first_string]

#     for i in range(length):
#         logits, previous_state = model.call(next_input, previous_state)
#         logits = np.array(logits[0,0,:])
#         top_n = np.argsort(logits)[-sample_n:]
#         n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
#         out_index = np.random.choice(top_n,p=n_logits)

#         text.append(reverse_vocab[out_index])
#         next_input = [[out_index]]

#     print(" ".join(text))


# def main():

#     train_list, test_list, dict = get_data('../../data/train.txt', '../../data/test.txt')

#     my_model = Model(len(dict))
    
#     train(my_model, train_list[0:len(train_list)-1], train_list[1:len(train_list)])
#     print(test(my_model, test_list[0:len(test_list)-1], test_list[1:len(test_list)]))

# if __name__ == '__main__':
#     main()