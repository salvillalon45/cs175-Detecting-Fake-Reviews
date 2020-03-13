# ---------------------------------------------------------------------------------------
#
# Name: src/classification.py
# Authors: Salvador Villalon
# Description: This script provides helper functions to create each of the following
#   classifiers:
#     - Logistic Regression Classifier
#     - Naïve Bayes Classifier (not used)
#     - K Nearest Neighbors Classifier
#     - Decision Tree Classifier
#     - Random Forest Classifier
#
# ---------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import functions


def logistic_regression(X, Y, test_fraction=0.25):
    classifier = linear_model.LogisticRegression(penalty='l2', fit_intercept=True)
    classifier.fit(X, Y)
    return classifier


def naive_bayes(X, Y, test_fraction=0.25):
    classifier = MultinomialNB()
    classifier.fit(X, Y)
    return classifier


def knearest_neighbors(X, Y, test_fraction=0.25):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X, Y)
    return classifier


def decision_trees(X, Y, test_fraction=0.25):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, Y)
    return classifier


def random_forest(X, Y, test_fraction=0.25):
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X, Y)
    return classifier
