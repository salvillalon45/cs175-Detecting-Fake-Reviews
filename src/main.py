# ---------------------------------------------------------------------------------------
#
# Name: src/main.py
# Authors: Brandon Teran, Salvador Villalon, Andrew Self
# Description: This script is meant to bring together all of the project's functionality.
#   Here we parse the datasets, train the Bag of Words and Doc2Vec models, and train and
#   predict using our classifiers.
#
# ---------------------------------------------------------------------------------------
import embeddings
import classification
import yelp_parser
import numpy as np
import functions
from sklearn.model_selection import train_test_split


def add_length_review_feature(X, length_of_reviews):
    '''
    Since we're adding features for every review (i.e., we're adding more columns),
    we need to add the extra columns to the training and testing feature matrices.
    '''
    rows = X.shape[0]
    cols = X.shape[1] + 1
    total = (rows * cols)

    new_X = np.arange(total).reshape(rows, cols)

    for i in range(len(X)):
        review_length = length_of_reviews[i]
        review_vector = X[i]
        review_vector = np.append(review_vector, review_length)
        new_X[i] = review_vector

    return new_X


def get_train_lists(model, train_targets, train_regressors, review_lengths):
    X = []
    for i in range(len(train_targets)):
        X.append(model.infer_vector(train_targets[i]))

    train_x = np.asarray(X)
    train_y = embeddings.add_unique_labels(train_regressors)
    return train_x, train_y


def get_test_lists(model, test_targets, test_regressors):
    test_list = []
    for i in range(len(test_targets)):
        test_list.append(model.infer_vector(test_targets[i]))

    test_x = np.asarray(test_list)
    test_y = embeddings.add_unique_labels(test_regressors)
    return test_x, test_y


def run_classifiers_with_doc2vec(reviews, scores, review_lengths, with_features=False):
    '''Corpus should be an array of TaggedDocument objects.'''
    corpus = list(embeddings.get_corpus(reviews, scores))[:20000]
    train_corpus, test_corpus = train_test_split(corpus, test_size=0.25, random_state=42)

    doc2vec_model = embeddings.create_doc2vec_model(train_corpus)
    train_targets, train_regressors = zip(*[(doc.words, doc.tags[0]) for doc in train_corpus])
    test_targets, test_regressors = zip(*[(doc.words, doc.tags[0]) for doc in test_corpus])

    '''
    For every review, we apply doc2vec_model.infer_vector(review). This creates
    a feature vector for every document (in our case, review) in the corpus.
    '''
    train_x, train_y = get_train_lists(doc2vec_model, train_targets, train_regressors, review_lengths)
    test_x,  test_y  = get_test_lists(doc2vec_model, test_targets, test_regressors)

    '''
    When the 'with_features' parameter=True, we add our extra features to the
    existing feature matrix.
    '''
    if with_features:
        prp_list = functions.create_pos_features(reviews)
        train_x = functions.add_pos_feature(train_x, prp_list)
        train_x = add_length_review_feature(train_x, review_lengths)
        test_x = functions.add_pos_feature(test_x, prp_list)
        test_x = add_length_review_feature(test_x, review_lengths)

    logistic_reg = classification.logistic_regression(train_x, train_y)
    k_nearest_n  = classification.knearest_neighbors(train_x, train_y)
    decision_trees = classification.decision_trees(train_x, train_y)
    random_forest = classification.random_forest(train_x, train_y)

    classifiers = [logistic_reg, k_nearest_n, decision_trees, random_forest]

    for i in range(len(classifiers)):
        print("-------------------------------------------------")
        if i == 0:
            print("Logistic Regression\n")
        if i == 1:
            print("K Nearest Neighbors\n")
        if i == 2:
            print("Decision Trees\n")
        if i == 3:
            print("Random Forest\n")

        '''Train and predict on classifiers[i] for both training and testing data.'''
        functions.train_classifier_and_evaluate_accuracy_on_training_data(classifiers[i], train_x, train_y)
        functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifiers[i], test_x, test_y)
        print('\n\n')


def run_classifiers_with_bow(reviews, scores, review_lengths, with_features=False):
    X, vectorizer = functions.create_bow_from_reviews(reviews)
    train_x, test_x, train_y, test_y = train_test_split(X, scores, test_size=0.25, random_state=42)

    '''
    When the 'with_features' parameter=True, we add our extra features to the
    existing feature matrix.
    '''
    if with_features:
        '''Create 'Part of Speech' feature vector for each review'''
        prp_list = functions.create_pos_features(reviews)
        '''Add both the POS and Review Length vectors to features'''
        train_x = functions.add_pos_feature(train_x, prp_list)
        train_x = add_length_review_feature(train_x, review_lengths)
        '''Do the same for the testing features'''
        test_x = functions.add_pos_feature(test_x, prp_list)
        test_x = add_length_review_feature(test_x, review_lengths)

    '''Create each classifier with Training Features and Training Labels.'''
    logistic_reg = classification.logistic_regression(train_x, train_y)
    k_nearest_n  = classification.knearest_neighbors(train_x, train_y)
    decision_trees = classification.decision_trees(train_x, train_y)
    random_forest = classification.random_forest(train_x, train_y)

    classifiers = [logistic_reg, k_nearest_n, decision_trees, random_forest]

    for i in range(len(classifiers)):
        print("-------------------------------------------------")
        if i == 0:
            print("Logistic Regression\n")
        if i == 1:
            print("K Nearest Neighbors\n")
        if i == 2:
            print("Decision Trees\n")
        if i == 3:
            print("Random Forest\n")

        '''Train and predict on classifiers[i] for both training and testing data.'''
        functions.train_classifier_and_evaluate_accuracy_on_training_data(classifiers[i], train_x, train_y)
        functions.train_classifier_and_evaluate_accuracy_on_testing_data(classifiers[i], test_x, test_y)
        print('\n\n')


if __name__ == '__main__':
    '''First parse the Yelp review data, into reviews, scores and length of each review.'''
    yelp_reviews, scores, review_lengths = yelp_parser.get_chi_hotel_review_score_list()
    run_classifiers_with_doc2vec(yelp_reviews, scores, review_lengths)
    run_classifiers_with_bow(yelp_reviews, scores, review_lengths)
    run_classifiers_with_doc2vec(yelp_reviews, scores, review_lengths, with_features=True)
