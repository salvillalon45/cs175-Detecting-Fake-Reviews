# --------------------------------------------------------------------------------------
#
# Name: src/yelp/with_features_main_bow.py
# Description:
# This file runs the classifiers for the Yelp dataset with no features added
#
# --------------------------------------------------------------------------------------
import functions
import classification
import yelp_parser

if __name__ == '__main__':
    print('Running Classifiers for Yelp chicago dataset')
    print("Does not include extra features")
    print("Using Bag of Words")
    print('------------------------------------------')

    reviews, scores, length_of_reviews = yelp_parser.parse_yelp_reviews()
    X, vectorizer = functions.create_bow_from_reviews(reviews)

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
