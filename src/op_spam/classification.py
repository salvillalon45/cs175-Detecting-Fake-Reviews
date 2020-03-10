# --------------------------------------------------------------------------------------
#
# Name: src/op_spam/classification.py
# Description:
# This file has the sklearn implmentation of the following classifiers: logistic regression,
# naive bayes, K nearest neighbor, decision trees, and random forests
#
# --------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import functions


def logistic_regression(X, Y, test_fraction=0.25):
    print(" ")
    print("Logistic Regression Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = linear_model.LogisticRegression(penalty='l2', fit_intercept=True)
    classifier.fit(X_train, Y_train)

    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)

    return classifier


def naive_bayes(X, Y, test_fraction=0.25):
    print(" ")
    print("Naive Bayes Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)
    print('Y TEST: ', Y_test)
    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)

    return classifier


def knearest_neighbors(X, Y, test_fraction=0.25):
    print(" ")
    print("K Nearest Neighbors Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X, Y)

    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)

    return classifier


def decision_trees(X, Y, test_fraction=0.25):
    print(" ")
    print("Decision Trees Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X, Y)

    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)

    return classifier


def random_forest(X, Y, test_fraction=0.25):
    print(" ")
    print("Decision Trees Test")
    print("-------------------------------------------------")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X, Y)

    functions.train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train)
    functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test)

    return classifier

