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
        scores.append(file_flag)
    else:
        scores.append(file_flag)

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
        scores.append(file_flag)
    else:
        scores.append(file_flag)

    reviews.append(review)


print(reviews)
print(scores)



vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", min_df=0.01)

# create a sparse BOW array from 'text' using vectorizer
X = vectorizer.fit_transform(text)

# an alternative above would be to use TfIDF rather than counts - which is very simple to do (but not needed here)
print('Data shape: ', X.shape)
