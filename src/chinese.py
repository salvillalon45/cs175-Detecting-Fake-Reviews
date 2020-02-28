# Module to use chinese data
import csv
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import os
import functions
from sklearn.feature_extraction.text import CountVectorizer


def read_chinese():
    print('Reading Chinese')
    file_name = '../datasets/data-hauyi/ICDM_REVIEWS_TO_RELEASE_encoding=utf-8.csv'

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

        line = line.split(',', maxsplit=5) # max split equals 5 so as to not split
        # print(line)
        label = line[1]
        review = line[5]
        # print(label)
        # print(review)
        if label == '+':
            labels.append(0)
        else:
            labels.append(1)
        reviews.append(review)

        if count > 5:
            break

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

    for i in reviews:
        print(i)
        s = seg.segment(i)
        print(s)
        segmented.append(s)
        # print('Tokenize: ')
        # print(seg.tokenize(s))
        count = count + 1
        if count > 5:
            break
    
    return(segmented)

def chinese_BOW(reviews):
    # vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", min_df=0.01)
    vectorizer = CountVectorizer()
    print(reviews)
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer

if __name__ == '__main__':
    labels, reviews = read_chinese()
    # print(labels,reviews)
    reviews = segment(labels, reviews)
    print(reviews)
    BOW, vec = chinese_BOW(reviews)
    print('BOW: ')
    print(BOW)
    print('BOW to array: ')
    print(BOW.toarray())
    print('shape: ')
    print(BOW.shape)
    print('get feature names')
    print(vec.get_feature_names())
    print('get params')
    print(vec.get_params())
    print('get stop words')
    print(vec.get_stop_words())

