import argparse
import os
import numpy as np
import tensorflow as tf
from loadIBC import UNK_TOKEN, get_data, batch
from matplotlib import pyplot as plt
from model import MLP
	
def train(model, train_inputs, train_labels, test_inputs, test_labels, build=False):
    """
    Trains model.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) 
    :param train_labels: train labels (all labels for training) 
    :param test_inputs: test inputs (all inputs for testing)
    :param test_labels: test labels (all labels for testing)
    :return: None
    """
    #TODO: Fill in
    num_batches = len(train_inputs) // model.batch_size
    losses = []
    accuracies = []

    inputs = train_inputs
    labels = train_labels
    for num_epoch in range(20):
        print("Num epoch: " + str(num_epoch))
        total_loss = 0
        for batch_num in range(num_batches):
            curr_batch = batch(inputs, model.batch_size, batch_num, len(inputs))
            curr_batch_labels = batch(labels, model.batch_size, batch_num, len(inputs))

            with tf.GradientTape() as tape:
                probs = model.call(curr_batch)
                loss = model.loss(probs, curr_batch_labels)
                
            total_loss += loss
            gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

			# Uncomment if you want to see progress while training.
            # if (batch_num % 100 == 0):
            #     print("Batch num " + str(batch_num) + " validation: " + str(test(model, test_inputs, test_labels)))

        accuracies.append(test(model, test_inputs, test_labels))
        losses.append(total_loss / num_batches)
    
    visualize_loss(losses)
    visualize_accuracies(accuracies)

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) 
    :param test_labels: train labels (all labels for testing)
    :returns: average accuracy
    """

    num_batches = len(test_inputs) // model.batch_size
    total_accuracy = 0

    for batch_num in range(num_batches):

        curr_batch = batch(test_inputs, model.batch_size, batch_num, len(test_inputs))
        curr_batch_labels = batch(test_labels, model.batch_size, batch_num, len(test_inputs))

        probs = model.call(curr_batch)
        predictions = np.argmax(probs, axis=1)
        total_accuracy += np.mean(predictions == curr_batch_labels)

    return total_accuracy / num_batches

def parseArguments():
	"""
	Parses terminal line arguments.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_weights", default=False, action="store_true")
	parser.add_argument("--load_weights", default=None, help="Filepath to weights.")
	parser.add_argument("--test_sentence", default=None, help="A sentence you want to test.")
	return parser.parse_args()

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per epoch.

	:param losses: list of losses for each epoch
	:return: None
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()

def visualize_accuracies(accuracies):
    """
    Uses Matplotlib to visualize loss per accuracy.

	:param accuracies: list of accuracies for each epoch
	:return: None
    """
    x = np.arange(1, len(accuracies)+1)
    plt.xlabel('i\'th Epoch')
    plt.ylabel('Accuracy Value')
    plt.title('Accuracy per Batch')
    plt.plot(x, accuracies)
    plt.show()

def save_model_weights(model):
	"""
	Save trained MLP model weights.

	:param model: model whose weights to save
	:return: None
	"""
	model_flag = "MLP"
	output_dir = os.path.join("model_ckpts", model_flag)
	output_path = os.path.join(output_dir, model_flag)
	os.makedirs("model_ckpts", exist_ok=True)
	os.makedirs(output_dir, exist_ok=True)
	model.save_weights(output_path)

def build(model, train_inputs, train_labels):
	'''
	Used when loading weights - the model must have at least one call so that
	variables exist.

	:param model: model to be built
	:param train_inputs: training data
	:param train_labels: train labels
	:return: None
	'''
	
	with tf.GradientTape() as tape:
		probs = model.call(train_inputs)
		loss = model.loss(probs, train_labels)
	gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
	model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def main(args):
	# Run normally: py main.py
	# Train, test, and save weights: py main.py --save_weights
	# Load, test weights: py main.py --load_weights model_ckpts/MLP/MLP
	# Load, test sentnece: py main.py --load_weights model_ckpts/MLP/MLP --test_sentence 'my test sentence'
	# Preprocess data.
	print("Running preprocessing...")
	train_data_phrases, test_data_phrases, train_data_labels, test_data_labels, vocab = get_data('data/train', 'data/test', 'data/train_labels', 'data/test_labels')
	print("Preprocessing complete.")

	my_MLP = MLP(len(vocab))

	if args.load_weights is None:
		# train if no weights are loaded
		print("Now training...")
		train(my_MLP, train_data_phrases, train_data_labels, test_data_phrases, test_data_labels)
		print("Training complete.")
	else:
		# else skip training and load weights
		print("Now loading model weights ...")
		build(my_MLP, train_data_phrases[0:10], train_data_labels[0:10])
		my_MLP.built = True
		my_MLP.load_weights(args.load_weights, by_name=False)
		print("Model weights loaded.")

	# test model
	# if args.test_sentence is None:
	# 	print("Now testing...")
	# 	print("Final accuracy: " + str(test(my_MLP, test_data_phrases, test_data_labels)))
	# else:
	# 	#else, test sentence given
	# 	test_phrase = [args.test_sentence.lower().split()]
	# 	test_ids = np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in test_phrase])
	# 	probs = my_MLP.call(test_ids).numpy()
	# 	print("The phrase " + args.test_sentence + " is " + str(probs[0, 0] * 100)[0:4] + "% liberal, " + str(probs[0, 1] * 100), "% conservative, and " + str(probs[0, 2] * 100) + "% neutral.")

	if args.save_weights:
		# save weights if flag is set
		print("Now saving model weights...")
		save_model_weights(my_MLP)
		print("Model weights saved.")
	sentences = [['liberal'], ['free market'], ['women in STEM'], ['conservative']]
	my_MLP.plot_similarities(vocab, sentences)

if __name__ == '__main__':
	args = parseArguments()
	main(args)