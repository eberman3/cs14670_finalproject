from loadIBC import get_data
from model import MLP, train, test

WINDOW_SIZE = 12


def main():

	print("Running preprocessing...")
	train_data_phrases, test_data_phrases, train_data_labels, test_data_labels, vocab, padding_index = get_data('ibcData.pkl')
	print("Preprocessing complete.")

	#model = Transformer_Seq2Seq(*model_args) 

	#train(model, train_french, train_english, eng_padding_index)

	#test(model, test_french, test_english, eng_padding_index)

	print("Now training...")
	my_MLP = MLP(len(vocab), WINDOW_SIZE)
	train(my_MLP, train_data_phrases, train_data_labels)
	print("Training complete.")

	print("Now testing...")
	print("Final accuracy: " + str(test(my_MLP, test_data_phrases, train_data_labels)))



	pass

if __name__ == '__main__':
	main()