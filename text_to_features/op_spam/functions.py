import os
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

import json as simplejson
import sklearn
from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_auc_score

from sklearn import linear_model
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt


def create_reviews_scores_arrays():
    # This function uses the op_spam dataset and extracts the reviews and the flag from the file name
    # This is needed so that we can do vectorization. We plan on using CountVectorization

    reviews = list()
    scores = list()
    # negative_polarity directory
    # ---------------------------------------------------------
    files_in_directory_negative_polarity = os.listdir("../../datasets/op_spam_v1.4/test_op_spam_v1.4/negative_polarity/")
    file_path_negative_polarity = "../../datasets/op_spam_v1.4/test_op_spam_v1.4/negative_polarity/"
    
    # Loop over the files in negative_polarity directory
    # Open the file line by line
    for file_name in files_in_directory_negative_polarity:
        print("The file is:: ", file_name)
        print("File flag:: ", file_name[0])
        
        file_flag = file_name[0]
        file_path = file_path_negative_polarity + file_name
        file_open = open(file_path)
        review = file_open.readline()
        
        print("The Review is:: ")
        print(review)
        print(" ")
        print("The File Flag is:: ")
        print(file_flag)
        
        if file_flag == "d":
            scores.append(0)
        else:
            scores.append(1)
    
        reviews.append(review)
    
    # positive_polarity directory
    # ---------------------------------------------------------
    files_in_directory_positive_polarity = os.listdir("../../datasets/op_spam_v1.4/test_op_spam_v1.4/positive_polarity/")
    file_path_positive_polarity = "../../datasets/op_spam_v1.4/test_op_spam_v1.4/positive_polarity/"
    
    # Loop over the files in positive_polarity directory
    # Open the file line by line
    for file_name in files_in_directory_positive_polarity:
        print("The file is:: ", file_name)
        print("File flag:: ", file_name[0])
        
        file_flag = file_name[0]
        file_path = file_path_positive_polarity + file_name
        file_open = open(file_path)
        review = file_open.readline()

        print("The Review is:: ")
        print(review)

        if file_flag == "d":
            scores.append(0)
        else:
            scores.append(1)
    
        reviews.append(review)
    
    return reviews, scores
    

def create_bow_from_reviews(reviews, scores):
    # Creating a bag of words by counting the number of times each word appears in a document.
    # This is possible using CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", min_df=0.01)
    
    # create a sparse BOW array from 'text' using vectorizer
    X = vectorizer.fit_transform(reviews)
    
    

reviews,scores = create_reviews_scores_arrays()
print(reviews)
print(scores)