# Module to use chinese data
import csv

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
        count = count + 1
        if count == 0:
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

    # print(len(labels), len(reviews))
    # print(labels)
    return labels, reviews

def chinese_BOW(labels, reviews):
    print('Creating BOW')



if __name__ == '__main__':
    labels, reviews = read_chinese()
    chinese_BOW(labels, reviews)