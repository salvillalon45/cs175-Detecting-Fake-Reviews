
# ---------------------------------------------------------------------------------------
# THIS VERSION OF THE CODE CONTAINS THE SOLUTION CODE TO THE ASSIGNMENT 

# ---------------------------------------------------------------------------------------
# CS 175, WINTER 2020: ASSIGNMENT 1
#
# The goal of this assignment is to give you some practice with manipulating text data, 
# including tokenization and creation of a sparse bag-of-words array for a set of documents,
# as well as building a simple logistic classifier and looking at which words get large
# positive and negative weights on a dataset of 20,000 reviews from Yelp
# 
# You should install Anaconda with Python 3.7 before starting this assignment
#
# General notes
#	- do not remove any code that is already provided: add your own code where appropriate
#	- add comments inline in the text to explain what you are doing 
#	- feel free to add error checking if you wish (but we will not grade you on this) 
#	- when you are done submit a copy of your edited version of this file, as Assignment1.py
#   - be sure to test your code on some simple examples before you submit it
#
# Grading
#   - problems 1 through 6 are each worth 20 points
#   - points will be deducted if 
# 		- the code does not execute 
#       - the code does not return the correct answers on simple test cases, 
#		- if the code is not general and only works on special cases, 
#		- if there are very few or no comments.
# 
# ---------------------------------------------------------------------------------------


# NOTE: for this assignment you will need to import the following libraries/modules
# All of these should be installed on your system if you have the latest version of Anaconda installed
import nltk 
from nltk import word_tokenize
import simplejson as json
import sklearn
from sklearn.feature_extraction.text import * 
from sklearn.model_selection import train_test_split 

from sklearn import linear_model 
from sklearn import metrics 

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------------
# PROBLEM 1
# Complete the definition of the function below so that it can take in a string
# and return the percentage of alphabetical characters in a string that match a particular
# alphabetical letter, where matching is not case-sensitive. The percentage is computed
# relative to the number of alphabetic characters in the string (so numbers, punctuation,
# white space, and all other non-alphabetic characters are ignored).
#
# NOTE: please read Section 3 of Chapter 1 in the NLTK online book to understand how to use the FreqDist method 
# ---------------------------------------------------------------------------------------
def letter_percentage(text ,letter):
	"""
	Parameters:
	text: string
	letter: a single alphabetical character (in lower case, e.g., 'a', 'b', ...)
	
	Returns:
	percentage: the percentage of alphabetical characters in <text> that match <letter>
		  where a match is defined irrespective of the case of the characters in <text>
				
	Example:
	letter_percentage('This is a cat.','t')  returns 20.0
	""" 
 
     # extract a list of alphabetic characters and convert to lower case
	charlist = [char.lower() for char in text if char.isalpha()]

     # create an fdist object for the list of lower case characters
	fdist = nltk.FreqDist(charlist)

     # calculate the frequency of the specific character "letter"
	frequency = fdist.freq(letter)
 
     # convert the frequency to a percentage
	character_percent = 100 *frequency
	p = '{0:.2f}'.format(character_percent)
	print('\nPercentage = ' ,p ,' of characters match the character' ,letter)
	return character_percent


# ---------------------------------------------------------------------------------------
# PROBLEM 2
# Complete the definition of the function below so that it can take as input either
# (a) a string or (b) a list of tokens of type nltk.text.Text
# convert the string to word tokens, run the NLTK part of speech parser on the word tokens
# using the 'universal' tagset, print out to the screen the percentage of tokens in the
# that correspond to each type of tag, and return a list of pairs of tokens and tags.
# 
# NOTE:
# 	- please read Section 1 of Chapter 5 in the NLTK online book for information about part of speech tagging
# 	- for word tokenization use the NLTK word_tokenize function with default settings
#  	- Print out the tags in order of decreasing frequency of occurrence
#	- Percentages printed out should be formatted to 2 decimal places of precision
# ---------------------------------------------------------------------------------------
def parts_of_speech(s):
	"""
	Parameters:
	s: input text as a string 
	
	Returns:
	The list of tokens and their POS tags from the string s, as a list of sublists 
	Prints out the total number of tokens and percentage of tokens with each tag 
				
	Example:
	s = 'This is a sentence. And this is a second sentence! Cool.'
	z1, z2 = parts_of_speech(s)  
		Total number of tokens is 14
		Tag: DET           Percentage of tokens =  28.57
		Tag: .             Percentage of tokens =  21.43
		Tag: NOUN          Percentage of tokens =  21.43
		....
	""" 
	 
	# tokenize the string into word tokens
	tokens = nltk.word_tokenize(s)
	
    # extract POS tags using the universal tagset with the NLTK POS tagger
	tokens_and_tags = nltk.pos_tag(tokens ,'universal')
		
	# Compute and print the total number of tokens  
	n = len(tokens_and_tags)
	print('Total number of tokens is' ,n)
	
	# extract the 2nd item from tokens_and_tags from each sublist
	# where item[1] is the 2nd item on each sublist
	tags = [ item[1] for item in tokens_and_tags ]  
	
	# count how often each of the tags occurs using FreqDist (from NLTK)
	tag_counts = nltk.FreqDist(tags)
 
     # sort the tag counts by frequency (using one of FreqDist's built in methods)
	sorted_tag_counts = tag_counts.most_common( len(tag_counts) )
	
     # print out each tag and the percentage of tokens associated with it, in descending order 
	for item in sorted_tag_counts:
		tag_percent = 100 * item[1 ] /n
		p = '{0:.2f}'.format(tag_percent)
		print('Tag:' ,item[0] ,'\t   Percentage of tokens = ', p )
  
	return( tokens_and_tags  )
	
	

# ---------------------------------------------------------------------------------------
# PROBLEM 3
# Complete the definition of the function below so that it 
# - reads in the file review_subset.json (using json.load)
# - extracts the text of the kth review
# - runs the parts_of_speech function (Problem 2) to compute the percentages of tokens for each part of speech
# ---------------------------------------------------------------------------------------
def review_pos(k ,filename):
	
	# print('\nLoading the file: review_subset.json\n')
	# name = 'review_subset.json'  # load data
	print('\nLoading the file: \n', filename) 
	with open(filename, 'r') as jfile:
		data = json.load(jfile)
	print('\nTotal number of reviews extracted =', len(data) )
 
	print('\nComputing the percentages for each part-of-speech for review' ,k)
	
	d = data[ k -1]  # extract the kth review (indexed from 0)
	s = d['text']  # extract text string associated with kth review
	print('Text from review ' ,k, ' is:')
	print(s)
	parts_of_speech(s)
		

# ---------------------------------------------------------------------------------------
# PROBLEM 4
# Create a bag of words (BOW) representation from text documents, using the Vectorizer function in scikit-learn
#
# The inputs are 
#  - a filename (you will use "yelp_reviews.json") containing the reviews in JSON format 
#  - the min_pos and max_neg parameters
#  - we label all reviews with scores > min_pos = 4 as "1"  
#  - we label all reviews with scores < max_neg = 2 as "0" 
#  - this creates a simple set of labels for binary classification, ignoring the neutral (score = 3) reviews
# 
#  The function extracts the text and scores for each review from the JSON data
#  It then tokenizes and creates a sparse bag-of-words array using scikit-learn vectorizer function
#  The number of rows in the array is the number of reviews with scores <=2 or >=4
#  The number of columns in the array is the number of terms in the vocabulary
#
#  NOTE: 
#  - please read the scikit-learn tutorial on text feature extraction before you start this problem:
#     https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction  
#  - in this function we will use scikit-learn tokenization (rather than NLTK)
# ---------------------------------------------------------------------------------------
def create_bow_from_reviews(filename, min_pos=4, max_neg=2): 
	
	print('\nLoading the file: \n', filename) 
	with open(filename, 'r') as jfile:
		data = json.load(jfile)
	print('\nTotal number of reviews extracted =', len(data) )

	text = []
	Y = []
	lengths = []
	print('\nExtracting tokens from each review.....(can be slow for a large number of reviews)......')   
	for d in data: 	# can substitute data[0:9] here if you want to test this function on just a few reviews 	
		review = d['text']     # keep only the text and label
		stars = int(d['stars'])
		if stars >= min_pos:   # represent scores > min_pos as "1"
			score = 1
		elif stars <= max_neg: # represent scores < max_neg as "0"
			score = 0
		else: # do not consider reviews with scores above max_neg and below min_pos (these reviews will be dropped)
			continue  
 
		text.append(review)   
		Y.append(score)
    
    # create an instance of a CountVectorizer, using 
    # (1) the standard 'english' stopword set 
    # (2) only keeping terms in the vocabulary that occur in at least 1% of documents
    # (3) allowing both unigrams and bigrams in the vocabulary (use "ngram_range=(1,2)" to do this)
	vectorizer = CountVectorizer(stop_words='english' ,min_df=0.01 ,ngram_range=(1 ,2))
	X = vectorizer.fit_transform(text)
	
	# an alternative above would be to use TfIDF rather than counts - which is very simple to do:
	# vectorizer = TfidfVectorizer(.... 
 
	print('Data shape: ', X.shape)
	
	# you can uncomment this next line if you want to see the full list of tokens in the vocabulary  
	# print('Vocabulary: ', vectorizer.get_feature_names())
 
	return X, Y, vectorizer
		 
		 
		 
# ---------------------------------------------------------------------------------------
# PROBLEM 5
#  Separate an X,Y dataset (X=features, Y=labels) into training and test subsets
#  Build a logistic classifier on the training subset
#  Evaluate performance on the test subset  
#
#  NOTE: before starting this problem please read the scikit-learn documentation on logistic classifiers:
#		https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
# ---------------------------------------------------------------------------------------		 
def logistic_classification(X, Y, test_fraction): 
	# should add comments defining what the inputs are what the function does
	
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
	#  set the state of the random number generator so that we get the same results across runs when testing our code
	 
	print('Number of training examples: ', X_train.shape[0])
	print('Number of testing examples: ', X_test.shape[0])   
	print('Vocabulary size: ', X_train.shape[1]) 
 

	# Specify the logistic classifier model
	classifier = linear_model.LogisticRegression(penalty='l2',  fit_intercept=True)  

	# Train a logistic regression classifier and evaluate accuracy on the training data
	print('\nTraining a model with', X_train.shape[0], 'examples.....')
	classifier.fit(X_train, Y_train) 
	train_predictions = classifier.predict(X_train)	 # Training
	train_accuracy = metrics.accuracy_score(Y_train, train_predictions)
	class_probabilities_train = classifier.predict_proba(X_train)
	train_auc_score = metrics.roc_auc_score(Y_train, class_probabilities_train[:, 1]);
	print('\nTraining:')
	print(' accuracy:' ,format( 100 *train_accuracy , '.2f') )
	print(' AUC value:', format( 100 *train_auc_score , '.2f') )

	# Compute and print accuracy and AUC on the test data
	print('\nTesting: ')
	test_predictions = classifier.predict(X_test)	 
	test_accuracy = metrics.accuracy_score(Y_test, test_predictions) 
	print(' accuracy:', format( 100 *test_accuracy , '.2f') )
	
	class_probabilities = classifier.predict_proba(X_test)
	test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities[:, 1]);
	print(' AUC value:', format( 100 *test_auc_score , '.2f') )
	
	return (classifier)
    

# ---------------------------------------------------------------------------------------
# PROBLEM 6
#   Takes as input
#     (1) a scikit-learn trained logistic regression classifier (e.g., trained in Problem 5) 
#     (2) a scikit-learn vectorizer object that produced the BOW features for the classifier
#   and prints out and returns
#   - the K terms in the vocabulary tokens with the largest positive weights  
#   - the K terms in the vocabulary with the largest negative weights 
#
# To write this code you will need to use the get_params() method for the logistic regression model 
# in scikit-learn, and you will also need to 
# ---------------------------------------------------------------------------------------				

def most_significant_terms(classifier, vectorizer, K):
	topK_pos_weights = classifier.coef_[0].argsort()[-K:][::-1]
	topK_neg_weights = classifier.coef_[0].argsort()[:K]
	word_list = vectorizer.get_feature_names()

     # Nothing written for this code yet. 
     # We could leave it blank and let students solve this however they want to
     # .... or we could provide some helper code. 
     #
     # Note that we will need to write a version of this code ourselves to grade the assignment
     # and to provide in the solutions we provide to the students after the assignments are submitted
     
     # ....
     
     # cycle through the positive weights, in the order of largest weight first and print out
     # K lines where each line contains 
     # (a) the term corresponding to the weight (a string)
     # (b) the weight value itself (a scalar printed to 3 decimal places)
	 
	print('Top K positive weight words:')
	for w in topK_pos_weights:
		print('%s : %.4f' % (word_list[w] ,classifier.coef_[0][w]))
		
	print('Top K negative weight words:')
	for w in topK_neg_weights:
		print('%s : %.4f' % (word_list[w] ,classifier.coef_[0][w]))
