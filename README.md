# Detecting_Semantic_Shift_In_Sarcasm
Final project for CSC2611 (Computational Models of Semantic Shift)

Below, the functionality of each file is explained. Generally each file only has 1 important function and does one important task.

- **tweet_scrapping.py** file is used to scrap tweets and save that in a pickle file.
- **generate_words_to_search.py** identifies high positive sentiment words and high negative sentiment words. A subset of these words will be used in training depending upon frequency of these words in the sarcastic and non-sarcastic corpus.
- **util.py** reads the pickle file and generates the cooccurrence matrix. It also provides pretrained glove embeddings for words, word2index dict and other utility functions.
  + tweets_pre_processing(...) function does the preprocessing for the tweets.
  + the Utils.get_data(...) method, generates the cooccurence matrix and word2index dict.
  + util.py has a member called words_to_search. It could be set to stopwords or sentiment words collected in generate_words_to_search.py
  + The main method of util.py can be run to collect cooccurrence matrix for context window sizes (10,15,20) and cooccurrence count function $1/d$, $1/(1+log_{10}(d))$ and $1/\sqrt{d}$. 
- **glove_transformed.py** generates the initialization embeddings by learning a transformation on the glove-twitter-50 vectors.
  + It trains the word-embeddings and also stores in in a pickle format that can be used by the word_vec_analysis.ipynb notebook 
- **glove_selective_training.py** trains the word vectors for the subset of words used in util.py
  + It trains the word-embeddings and also stores in in a pickle format that can be used by the word_vec_analysis.ipynb notebook
- **test_men.py** tests MEN dataset on the word-embeddings. It selects the examples from the dataset in which both of word-pairs are present in word2index. 
- **run_script.sh** is a convenience file to run both glove_transformed.py and glove_selective_training.py one after another.
- **notebooks/word_vec_analysis.ipynb** was used to collect the results and generate the plots.
