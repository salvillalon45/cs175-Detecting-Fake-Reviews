from sklearn import linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from classification import logisticRegression
from collections import defaultdict
import functions
import numpy as np
import gensim
from gensim.models import Doc2Vec
from nltk.corpus import stopwords


def get_corpus(reviews: [str], scores: [int]):
    stoplist = stopwords.words('english')
    review_tokens = []
    for review in reviews:
        review_tokens.append([word for word in review.lower().split() if word not in stoplist])
    for i, text in enumerate(review_tokens):
        yield gensim.models.doc2vec.TaggedDocument(text, [scores[i]])


def createDoc2VecModel(reviews, scores):
    corpus = list(get_corpus(reviews, scores))[:20000]
    train_corpus, test_corpus = train_test_split(corpus, test_size=0.25, random_state=42)
    print(len(train_corpus))
    model = Doc2Vec(window=100, dm=1, vector_size=50, min_count=2)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    train_targets, train_regressors = zip(*[(doc.words, doc.tags[0]) for doc in train_corpus])
    test_targets, test_regressors = zip(*[(doc.words, doc.tags[0]) for doc in test_corpus])

    X = []
    for i in range(len(train_targets)):
        X.append(model.infer_vector(train_targets[i]))

    train_x = np.asarray(X)
    print(train_x.shape)

    logreg = linear_model.LogisticRegression()
    logreg.fit(train_x, train_y)

    test_list = []
    for i in range(len(test_targets)):
        test_list.append(model.infer_vector(test_targets[i]))

    test_x = np.asarray(test_list)
    test_Y = np.asarray(test_regressors)
    test_y = labelEncoder.transform(test_Y)

    predictions = logreg.predict(test_x)
    np.mean(test_y)

    acc = sum(predictions == test_y) / len(test_y)
    print(acc)
