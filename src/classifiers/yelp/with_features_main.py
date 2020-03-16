# --------------------------------------------------------------------------------------
#
# Name: src/yelp/with_features_main_bow.py
# Description:
# This file runs the classifiers for the Yelp dataset with extra features added
#
# --------------------------------------------------------------------------------------
import functions
import classification
import yelp_parser

if __name__ == '__main__':
    print('Running Classifiers for Yelp Chicago dataset')
    print("Does include features")
    print("Using Bag of Words")
    print('------------------------------------------')

    reviews, scores, length_of_reviews = yelp_parser.parse_yelp_reviews()
    X, vectorizer = functions.create_bow_from_reviews(reviews, scores)

    # Adding length of a each review feature
    print("After adding length review feature")
    X = functions.add_length_review_feature(X, length_of_reviews)
    print(X)

    # Adding Part of Speech Tag Feature
    print("After adding Part of Speech Tag feature")
    prp_list = functions.create_pos_features(reviews)
    X = functions.add_pos_feature(X, prp_list)
    print(X)

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
