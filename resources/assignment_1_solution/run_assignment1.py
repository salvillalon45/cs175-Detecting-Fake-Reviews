 
from nltk import word_tokenize
import json as simplejson

import sklearn
from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split 

from sklearn import linear_model 
from sklearn import metrics 

import numpy as np
import matplotlib.pyplot as plt

# turn off annoying warnings
import warnings 
warnings.simplefilter('ignore')
 
# from assignment1_solution import *
from assignment1 import *


if __name__ == '__main__': 

    # test the letter_percentage function with simple input
    letter_percentage('This is a cat.','t')

    # test the parts_of_speech function with simple input
    parts_of_speech('This is a very simple test sentence to test the part of speech function in NLTK.')
    
    # load yelp reviews and compute percentages of parts of speech for the Kth review
    K = 1
    review_pos(K, 'yelp_reviews.json')
    
    # read in the review text and tokenize the text in each review
    X, Y , vectorizer_BOW = create_bow_from_reviews('yelp_reviews.json')  
    
    # run a logistic classifier on the reviews, specifying the fraction to be used for testing  
    test_fraction = 0.8
    logistic_classifier = logistic_classification(X, Y,test_fraction)  
    
    # print out and return the most significant positive and negative weights (and associated terms) 
    most_significant_terms(logistic_classifier, vectorizer_BOW, K=10)