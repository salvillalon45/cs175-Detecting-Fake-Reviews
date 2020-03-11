# --------------------------------------------------------------------------------------
#
# Name: src/op_spam/op_spam_util.py
# Description:
# This file has the utility functions needed to add more features and parse the data
#
# --------------------------------------------------------------------------------------
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np


def parse_op_spam():
    """
    This function uses the op_spam dataset and extracts the reviews, flag (d or t), and length of each review from the file name
    """
    print("Inside parse_op_spam()")

    reviews = list()
    scores = list()
    length_of_reviews = list()

    # negative_polarity directory
    # ---------------------------------------------------------
    files_in_directory_negative_polarity = os.listdir("../../datasets/op_spam_v1.4/test_op_spam_v1.4/negative_polarity/")
    file_path_negative_polarity = "../../datasets/op_spam_v1.4/test_op_spam_v1.4/negative_polarity/"
    # src / classifiers / op_spam / op_spam_util.py
    # Loop over the files in negative_polarity directory
    # Open the file line by line
    for file_name in files_in_directory_negative_polarity:

        file_flag = file_name[0]
        file_path = file_path_negative_polarity + file_name
        file_open = open(file_path)
        review = file_open.readline()

        if file_flag == "d":
            scores.append(0)
        else:
            scores.append(1)

        reviews.append(review)
        length_of_reviews.append(len(review))

    # positive_polarity directory
    # ---------------------------------------------------------
    files_in_directory_positive_polarity = os.listdir("../../datasets/op_spam_v1.4/test_op_spam_v1.4/positive_polarity/")
    file_path_positive_polarity = "../../datasets/op_spam_v1.4/test_op_spam_v1.4/positive_polarity/"

    # Loop over the files in positive_polarity directory
    # Open the file line by line
    for file_name in files_in_directory_positive_polarity:

        file_flag = file_name[0]
        file_path = file_path_positive_polarity + file_name
        file_open = open(file_path)
        review = file_open.readline()

        if file_flag == "d":
            scores.append(0)
        else:
            scores.append(1)

        reviews.append(review)
        length_of_reviews.append(len(review))

    return reviews, scores, length_of_reviews


def create_bow_from_reviews(reviews):
    """
    Creating a bag of words by counting the number of times each word appears in a document.
    This is possible using CountVectorizer
    """
    print("Inside create_bow_from_reviews()")

    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", min_df=0.01)

    # create a sparse BOW array from 'text' using vectorizer
    X = vectorizer.fit_transform(reviews)

    return X, vectorizer


def create_pos_features(reviews):
    """
    This function creates a frequencey list of PRP tags for each review.
    PRP means personal pronoun. It is said that Fake reviews usually use PRP a lot.
    This includes words like Me or I
    """
    print("Inside create_pos_features()")

    prp_list = list()

    for review in reviews:
        tokens = nltk.word_tokenize(review)
        pos_list = nltk.pos_tag(tokens)
        prp_count = 0

        for pos in pos_list:
            tag = pos[1]

            if tag == "PRP":
                prp_count = prp_count + 1

        prp_list.append(prp_count)

    return prp_list


def add_length_review_feature(X, length_of_reviews):
    """
    This function adds the length of each review to the vector of each review
    """
    print("Inside add_length_review_feature()")

    rows = X.shape[0]
    cols = X.shape[1] + 1
    total = (rows * cols)

    new_X = np.arange(total).reshape(rows, cols)

    for i in range(len(X.toarray())):
        review_length = length_of_reviews[i]
        review_vector = X.toarray()[i]
        review_vector = np.append(review_vector, review_length)

        new_X[i] = review_vector

    return new_X


def add_pos_feature(X, prp_list):
    """
    This function adds the PRP frequency to the vector of each review
    """
    print("Inside add_pos_feature()")

    rows = X.shape[0]
    cols = X.shape[1] + 1
    total = (rows * cols)

    new_X = np.arange(total).reshape(rows, cols)

    for i in range(len(X)):
        prp_count = prp_list[i]
        review_vector = X[i]
        review_vector = np.append(review_vector, prp_count)

        new_X[i] = review_vector

    return new_X


def train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train):
    """
    This function calculates the accuracy and AUC value for training data
    """
    train_predictions = classifier.predict(X_train)
    train_accuracy = metrics.accuracy_score(Y_train, train_predictions)

    class_probabilities_train = classifier.predict_proba(X_train)
    train_auc_score = metrics.roc_auc_score(Y_train, class_probabilities_train[:, 1])

    print("Training: ")
    print(" Accuracy: ", format(100 * train_accuracy, ".2f"))
    print(" AUC Value: ", format(100 * train_auc_score, ".2f"))


def train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test):
    """
    This function calculates the accuracy and AUC value for training data
    """
    test_predictions = classifier.predict(X_test)
    test_accuracy = metrics.accuracy_score(Y_test, test_predictions)

    class_probabilities = classifier.predict_proba(X_test)
    test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities[:, 1])

    print("\nTesting: ")
    print(" Accuracy: ", format(100 * test_accuracy, ".2f"))
    print(" AUC Values: ", format(100 * test_auc_score, ".2f"))


def most_significant_terms(classifier, vectorizer, K):
    topK_pos_weights = classifier.coef_[0].argsort()[-K:][::-1]
    topK_neg_weights = classifier.coef_[0].argsort()[:K]
    word_list = vectorizer.get_feature_names()

    # cycle through the positive weights, in the order of largest weight first and print out
    # K lines where each line contains
    # (a) the term corresponding to the weight (a string)
    # (b) the weight value itself (a scalar printed to 3 decimal places)

    print('Top K positive weight words:')
    for w in topK_pos_weights:
        print('%s : %.4f' % (word_list[w],classifier.coef_[0][w]))

    print('Top K negative weight words:')
    for w in topK_neg_weights:
        print('%s : %.4f' % (word_list[w],classifier.coef_[0][w]))
