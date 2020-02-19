# Functions to create various text classification models
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import functions
import yelp_parser

def logisticRegression(X, Y, test_fraction=0.25):
    print(" ")
    print("Logistic Regression Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = linear_model.LogisticRegression(penalty='l2', fit_intercept=True)
    classifier.fit(X_train, Y_train)
    
    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)
    

def naiveBayes(X, Y, test_fraction=0.25):
    print(" ")
    print("Naive Bayes Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)

    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)
    

def kNearestNeighbors(X, Y, test_fraction=0.25):
    print(" ")
    print("K Nearest Neighbors Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X, Y)
    
    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)
    
    
def decisionTrees(X, Y, test_fraction=0.25):
    print(" ")
    print("Decision Trees Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, Y)
    
    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)
    
    
def randomForest(X, Y, test_fraction=0.25):
    print(" ")
    print("Decision Trees Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X, Y)
    
    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)
    
    
if __name__ == '__main__':
    print('Running Program')
    print('------------------------------------------')
    # get BOW and ground truth from functions.py
    X, Y = functions.create_reviews_scores_arrays()
    X = functions.create_bow_from_reviews(X, Y)
    
    logisticRegression(X, Y)
    naiveBayes(X, Y)
    kNearestNeighbors(X,Y)
    decisionTrees(X, Y)
