# --------------------------------------------------------------------------------------
#
# Name: src/dianping/no_features_main_bow.py
# Description:
# This file runs the classifiers for the dianping dataset with no features added
#
# --------------------------------------------------------------------------------------
import classification
import chinese as dianping


if __name__ == '__main__':
    print('Running Classifiers for dianping dataset')
    print("Does not include extra features")
    print("Using Bag of Words")
    print('------------------------------------------')

    stop = dianping.gather_stopwords()
    labels, reviews = dianping.read_chinese()

    BOW, vec = dianping.chinese_BOW(reviews, stop)

    # Logistic Regression
    # --------------------------------------------
    classifier = classification.logistic_regression(BOW, labels)

    # Naive Bayes
    # --------------------------------------------
    classifier = classification.naive_bayes(BOW, labels)

    # K Nearest Neighbors
    # --------------------------------------------
    classifier = classification.knearest_neighbors(BOW, labels)

    # Decision Trees
    # --------------------------------------------
    classifier = classification.decision_trees(BOW, labels)

    # Random Forests
    # --------------------------------------------
    classifier = classification.random_forest(BOW, labels)
