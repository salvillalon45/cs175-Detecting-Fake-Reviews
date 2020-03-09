# --------------------------------------------------------------------------------------
#
# Name: src/op_spam/main.py
# Description:
# This file runs the classifiers for the op_spam dataset
#
# --------------------------------------------------------------------------------------
import functions
import classification

# functions.most_significant_terms(classifier, vectorizer, K=10)

if __name__ == '__main__':
    print('Running Classifiers for op_spam dataset')
    print('------------------------------------------')

    # get BOW and ground truth from utils.py
    reviews, scores, length_of_reviews = functions.create_reviews_scores_arrays()
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
