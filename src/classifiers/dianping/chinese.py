# Module to use chinese data
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import os
import sys
sys.path.append('../op_spam')
from sklearn.feature_extraction.text import CountVectorizer


def gather_stopwords():
    print('Gathering stop words')
    stop_words = []
    file_name = '../../datasets/data-hauyi/stopwords.txt'
    for line in open(file_name):
        line = line.strip()
        # print(line)
        stop_words.append(line)
    print(stop_words)
    return stop_words


def read_chinese():
    print('Reading Chinese')

    # file_name = '../datasets/data-hauyi/ICDM_REVIEWS_TO_RELEASE_encoding=utf-8.csv'
    file_name = '../../datasets/data-hauyi/reviews.txt'

    # reader = csv.reader(file_name, delimiter=',')

    # for row in reader:
    #     print(row)

    # for line in open('../requirements.txt'):
    #     print(line)

    labels = []
    reviews = []

    count = 0
    for line in open(file_name):
        print('HERE')
        # count = count
        count = count + 1
        if count == 1:
            continue

        # line = line.split(',', maxsplit=5) # max split equals 5 so as to not split
        line = line.split(' ', maxsplit=1) # max split equals 5 so as to not split

        # print(line)
        # label = line[1]
        # review = line[5]
        label = line[0]
        review = line[1]
        # print('label')
        # print('review')
        # print(label)
        # print(review)
        # if label == '+':
        if label == '0':
            labels.append(0)
        else:
            labels.append(1)
        reviews.append(review)

        if count > 9000:
            break
        print(count)

    # print(len(labels), len(reviews))
    # print(labels)
    return labels, reviews


def segment(labels, reviews):

    segmented = []

    print('Creating BOW')
    # seg = StanfordSegmenter('../../datasets/data-hauyi/stanford-segmenter-2018-10-16')
    os.environ["STANFORD_SEGMENTER"] = '../datasets/data-hauyi/stanford-segmenter-2018-10-16'
    seg = StanfordSegmenter('../datasets/data-hauyi/stanford-segmenter-2018-10-16/stanford-segmenter-3.9.2.jar')
    seg.default_config('zh',)
    count = 0

    file_out = open('reviews.txt','a+')

    for i in range(len(reviews)):
        # print(i)
        s = seg.segment(reviews[i])
        l = labels[i]
        # print(s)
        line = str(l) + ' ' + s
        file_out.write(line)
        segmented.append(s)
        # print('Tokenize: ')
        # print(seg.tokenize(s))
        count = count + 1
        # if count > 5:
        #     break
        print('Count: ', count)

    return(segmented)


def chinese_BOW(reviews, stop):
    # vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", min_df=0.01)
    # vectorizer = CountVectorizer()
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop, min_df=0.01)
    # print(reviews)
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer

def train_classifier_and_evaluate_accuracy_on_training_data(classifier, X_train, Y_train):
    """
    This function calculates the accuracy and AUC value for training data
    """
    train_predictions = classifier.predict(X_train)
    train_accuracy = metrics.accuracy_score(Y_train, train_predictions)

    class_probabilities_train = classifier.predict_proba(X_train)
    train_auc_score = metrics.roc_auc_score(Y_train, class_probabilities_train[:, 1])

    print("Training: ")
    print(" Accuracy: ", format(100 * train_accuracy, ".2f"))
    print(" AUC Value: ", format(100 * train_auc_score, ".2f"))

def train_classifier_and_evaluate_accuracy_on_testing_data(classifier, X_test, Y_test):
    """
    This function calculates the accuracy and AUC value for training data
    """
    test_predictions = classifier.predict(X_test)
    test_accuracy = metrics.accuracy_score(Y_test, test_predictions)

    class_probabilities = classifier.predict_proba(X_test)
    test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities[:, 1])

    print("\nTesting: ")
    print(" Accuracy: ", format(100 * test_accuracy, ".2f"))
    print(" AUC Values: ", format(100 * test_auc_score, ".2f"))
