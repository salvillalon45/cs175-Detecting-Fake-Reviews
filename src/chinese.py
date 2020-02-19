# Module to use chinese data
import csv
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import os


def read_chinese():
    print('Reading Chinese')
    file_name = '../../datasets/data-hauyi/ICDM_REVIEWS_TO_RELEASE_encoding=utf-8.csv'
    reader = csv.reader(file_name, delimiter=',')
    # for row in reader:
    #     print(row)

    labels = []
    reviews = []

    count = 0
    for line in open(file_name):
        count = count
        if count == 0:
            continue
        count = count + 1
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

    # print(len(labels), len(reviews))
    # print(labels)
    return labels, reviews



def chinese_BOW(labels, reviews):
    print('Creating BOW')
    # seg = StanfordSegmenter('../../datasets/data-hauyi/stanford-segmenter-2018-10-16')
    os.environ["STANFORD_SEGMENTER"] = '../../datasets/data-hauyi/stanford-segmenter-2018-10-16'
    seg = StanfordSegmenter('../../datasets/data-hauyi/stanford-segmenter-2018-10-16/stanford-segmenter-3.9.2.jar')
    seg.default_config('zh',)
    for i in reviews:
        print(i)
        print(seg.segment(i))
    



if __name__ == '__main__':
    labels, reviews = read_chinese()
    chinese_BOW(labels, reviews)