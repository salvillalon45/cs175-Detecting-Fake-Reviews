import gensim
import os
import numpy as np
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
import yelp_parser


def parse_yelp_reviews() -> ([str], [str]):
    return yelp_parser.get_chi_hotel_review_score_list()


def parse_opspam_reviews():
    """
    This function uses the op_spam dataset and extracts the reviews and the flag from the file name
    """
    print("Inside create_reviews_scores_arrays()")

    reviews = list()
    scores = list()
    length_of_reviews = list()

    # negative_polarity directory
    # ---------------------------------------------------------
    files_in_directory_negative_polarity = os.listdir("../../datasets/op_spam_v1.4/test_op_spam_v1.4/negative_polarity/")
    file_path_negative_polarity = "../../datasets/op_spam_v1.4/test_op_spam_v1.4/negative_polarity/"

    # Loop over the files in negative_polarity directory
    # Open the file line by line
    for file_name in files_in_directory_negative_polarity:

        file_flag = file_name[0]
        file_path = file_path_negative_polarity + file_name
        file_open = open(file_path)
        review = file_open.readline()

        if file_flag == "d":
            scores.append(0)
        else:
            scores.append(1)

        reviews.append(review)
        length_of_reviews.append(len(review))

    # positive_polarity directory
    # ---------------------------------------------------------
    files_in_directory_positive_polarity = os.listdir("../../datasets/op_spam_v1.4/test_op_spam_v1.4/positive_polarity/")
    file_path_positive_polarity = "../../datasets/op_spam_v1.4/test_op_spam_v1.4/positive_polarity/"

    # Loop over the files in positive_polarity directory
    # Open the file line by line
    for file_name in files_in_directory_positive_polarity:

        file_flag = file_name[0]
        file_path = file_path_positive_polarity + file_name
        file_open = open(file_path)
        review = file_open.readline()

        if file_flag == "d":
            scores.append(0)
        else:
            scores.append(1)

        reviews.append(review)
        length_of_reviews.append(len(review))

    return reviews, scores, length_of_reviews


def get_corpus(reviews: [str], scores: [int]):
    print("Inside get_corpus()")

    stoplist = stopwords.words('english')
    review_tokens = []

    for review in reviews:
        review_tokens.append([word for word in review.lower().split() if word not in stoplist])

    for i, text in enumerate(review_tokens):
        yield gensim.models.doc2vec.TaggedDocument(text, [scores[i]])


def train_model_from_corpus(reviews, scores):
    print("Inside train_model_from_corpus()")

    corpus = list(get_corpus(reviews, scores))[:20000]
    train_corpus, test_corpus = train_test_split(corpus, test_size=0.25, random_state=42)

    model = Doc2Vec(window=100, dm=1, vector_size=50, min_count=2)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    train_targets, train_regressors = zip(*[(doc.words, doc.tags[0]) for doc in train_corpus])
    test_targets, test_regressors = zip(*[(doc.words, doc.tags[0]) for doc in test_corpus])
    print("What is train_targets:: ", type(train_targets[0]))

    X = []
    for i in range(len(train_targets)):
        X.append(model.infer_vector(train_targets[i]))

    train_x = np.asarray(X)

    return model, train_targets, train_regressors, test_targets, test_regressors, train_x


def get_prediction(review, model, train_targets, train_regressors, test_targets, test_regressors, train_x):
    print("Inside get_prediction()")

    Y = np.asarray(train_regressors)
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(Y)
    train_y = label_encoder.transform(Y)
    np.mean(train_y)

    print("Review Before:: what is review:: ", type(review))
    # review = np.array(review).reshape(-1, 1)
    print("Review after:: what is review:: ", review)
    vector = model.infer_vector(review)
    print("what is vector:: ", vector)
    test_x = np.asarray([vector])
    test_Y = np.asarray(test_regressors)
    test_y = label_encoder.transform(test_Y)

    log_reg = linear_model.LogisticRegression()
    log_reg.fit(train_x, train_y)

    preds = log_reg.predict(test_x)
    print("Prediction is:: ", preds)
    np.mean(test_y)

    acc = sum(preds == test_y) / len(test_y)
    print("Accuracy:: ", acc)
