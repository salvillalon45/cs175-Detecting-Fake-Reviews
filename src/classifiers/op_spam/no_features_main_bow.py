# --------------------------------------------------------------------------------------
#
# Name: src/op_spam/with_features_main_bow.py
# Description:
# This file runs the classifiers for the op_spam dataset with no features added
#
# --------------------------------------------------------------------------------------
import op_spam_util as op_spam
import classification

# functions.most_significant_terms(classifier, vectorizer, K=10)

if __name__ == '__main__':
    print('Running Classifiers for op_spam dataset')
    print("Does not include extra features")
    print("Using Bag of Words")
    print('------------------------------------------')

    reviews, scores, length_of_reviews = op_spam.parse_op_spam()
    X, vectorizer = op_spam.create_bow_from_reviews(reviews)

    # Logistic Regression
    # --------------------------------------------
    classifier = classification.logistic_regression(X, scores)

    # Naive Bayes
    # --------------------------------------------
    classifier = classification.naive_bayes(X, scores)

    # K Nearest Neighbors
    # --------------------------------------------
    classifier = classification.knearest_neighbors(X, scores)

    # Decision Trees
    # --------------------------------------------
    classifier = classification.decision_trees(X, scores)

    # Random Forests
    # --------------------------------------------
    classifier = classification.random_forest(X, scores)
