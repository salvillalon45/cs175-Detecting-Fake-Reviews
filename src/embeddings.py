# ---------------------------------------------------------------------------------------
#
# Name: src/embeddings.py
# Authors: Brandon Teran
# Description: This script provides utility functions to create a corpus from the
#   reviews in the form of [TaggedDocument]. It also provides functions to create, train
#   and get a Doc2Vec embedding of said corpus.
#
# ---------------------------------------------------------------------------------------
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np
import gensim
from gensim.models import Doc2Vec
from nltk.corpus import stopwords


def get_corpus(reviews: [str], scores: [int]):
    '''
    Iterate over the reviews and corresponding scores and create a TaggedDocument
    object for each pair. These TaggedDocument objects make it easier to create Training
    and Testing matrices.
    '''
    stoplist = stopwords.words('english')
    review_tokens = []
    for review in reviews:
        review_tokens.append([word for word in review.lower().split() if word not in stoplist])
    for i, text in enumerate(review_tokens):
        yield gensim.models.doc2vec.TaggedDocument(text, [scores[i]])


def add_unique_labels(train_regressors):
    '''Go through the labels vector and give a unique ID to each label.'''
    Y = np.asarray(train_regressors)
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(Y)
    train_y = labelEncoder.transform(Y)
    return train_y


def create_doc2vec_model(train_corpus):
    model = Doc2Vec(window=100, dm=1, vector_size=50, min_count=2)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model
