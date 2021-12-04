from loadIBC import get_data

WINDOW_SIZE = 14

def main():

	print("Running preprocessing...")
	train_data_phrases, test_data_phrases, train_data_labels, test_data_labels = get_data('ibcData.pkl')
	print("Preprocessing complete.")

	#model = Transformer_Seq2Seq(*model_args) 

	#train(model, train_french, train_english, eng_padding_index)

	#test(model, test_french, test_english, eng_padding_index)

	pass

if __name__ == '__main__':
	main()