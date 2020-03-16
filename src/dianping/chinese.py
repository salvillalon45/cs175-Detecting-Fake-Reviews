# Module to use chinese data
import csv
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import os, sys
sys.path.append('../op_spam')
import functions
from sklearn.feature_extraction.text import CountVectorizer
import classification

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



if __name__ == '__main__':
    # print('Gathering stopwords')
    stop = gather_stopwords()
    labels, reviews = read_chinese()
    # print('labels: ', labels)
    # print('len label: ', len(labels))
    # print('reviews: ', reviews)
    # print('len reviews: ', len(reviews))
    # # print(labels,reviews)
    # # print(reviews)
    BOW, vec = chinese_BOW(reviews, stop)
    # # print('len bow: ', BOW.shape)
    # # print('BOW: ')
    # # print(BOW)
    # # print('BOW to array: ')
    # # print(BOW.toarray())
    # # print('shape: ')
    # # print(BOW.shape)
    # # print('get feature names')
    # # print(vec.get_feature_names())
    # # print('get params')
    # # print(vec.get_params())
    # # print('get stop words')
    # # print(vec.get_stop_words())

    logistic = classification.logistic_regression(BOW,labels)
    classification.naive_bayes(BOW, labels)
    classification.knearest_neighbors(BOW,labels)
    classification.decision_trees(BOW,labels)
    classification.random_forest(BOW,labels)

    # functions.most_significant_terms(logistic, vec, 10)

