# --------------------------------------------------------------------------------------
#
# Name: src/dianping/no_features_main_bow.py
# Description:
# This file runs the classifiers for the dianping dataset with no features added
#
# --------------------------------------------------------------------------------------
import functions
import classification


if __name__ == '__main__':
    print('Running Classifiers for dianping dataset')
    print("Does not include extra features")
    print("Using Bag of Words")
    print('------------------------------------------')

    reviews, scores, length_of_reviews = functions.create_reviews_scores_arrays()
    X, vectorizer = functions.create_bow_from_reviews(reviews, scores)

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
