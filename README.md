# billion-word-imputation
# Dataset
Billion Word Imputation, Kaggle
https://www.kaggle.com/c/billion-word-imputation

# Dependencies
## Standford CoreNLP for Python
pip install stanfordcorenlp
## Standford CoreNLP
https://stanfordnlp.github.io/CoreNLP/
## word2word.json file
It is too large to be included in Git. I will set up a download link later.

# Instructions
Clone this repository

Download [word2word.json](https://drive.google.com/file/d/1wS-kjyIA4e-gfT9yeYikS0SiBhLXVPTe/view) and save it in the same folder as of the code

The code is tested in Python 3.6.5

Then install stanfordcorenlp for python: pip install stanfordcorenlp

download [Stanford CoreNLP](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip)

extract Stanford CoreNLP in the same folder as of the code

It is not necessary to run training.py if you have download word2word.json.

Run predict.py to get a prediction result.

Submit the result to the [Kaggle competition](https://www.kaggle.com/c/billion-word-imputation).

[Our full test dataset result with weight 1-5-5-1](https://drive.google.com/open?id=1zOwllPLPUrd-UFNiI78_7M_W3xRgAO9x)

[Our full test dataset result with weight 1-1-1-1](https://drive.google.com/open?id=1n82Ue9NeC5b7eGRWL3JtcXRxKwOWvlaM)

# Parameters
In the name of predict output file, the four numbers represents:

α1: tag2tagweight_blank: The weight assigned to tag transition probability when identifying the blanks

β1: word2wordweight_blank: The weight assigned to word transition probability when identifying the blanks 
Then when choosing actual words

α2: tag2wordweight_word: The weight assigned to emission probability when choosing words 

β2: word2wordweight_word: The weight assigned to word transition probability when choosing words

predict_1_5_5_1.txt means prediction with α1=1, β1=0.5, α2=0.5, β2=1