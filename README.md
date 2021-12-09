README

Design Choices:
Preprocess: In preprocessing, we read in the data from pickled files. They are already in Python list format.
Model: Our model consists of an embedding layer and 5 dense layers sequentially. Inputs are phrases in ID form of different lengths, ex:
    [[300, 1],
     [120, 34, 4]]
Each ID gets turned into a 512 dimensional vector and then each row gets averaged to get batch_size * 512. Then that is passed through the
dense layers. Our highest accuracy is ~78%.
Main: In main is where our model is called. There are a number of things the user can do:

	Train, general test: py main.py
	Train, general test, and save weights: py main.py --save_weights
	Load weights, general test: py main.py --load_weights model_ckpts/MLP/MLP
	Load weights, test sentnece: py main.py --load_weights model_ckpts/MLP/MLP --test_sentence 'my test sentence'
    Load weights, plot similarities of sentences: py main.py --load_weights model_ckpts/MLP/MLP --plot_similarities
        - in addition, if you want to change sentences, change the line 176 to be:
            sentences = [['test sentence 1'], ['test sentence 2']] etc.

Bugs: No known bugs.