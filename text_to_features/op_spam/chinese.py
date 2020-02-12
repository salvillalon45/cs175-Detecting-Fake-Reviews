# Module to use chinese data
import csv

def read_chinese():
    print('Reading Chinese')
    file_name = '../../datasets/data-hauyi/ICDM_REVIEWS_TO_RELEASE_encoding=utf-8.csv'
    reader = csv.reader(file_name, delimiter=',')
    # for row in reader:
    #     print(row)

    labels = []
    text = []

    count = 0
    for line in open(file_name):
        count = count + 1
        if count > 3:
            break
        line = line.split(',', maxsplit=5) # max split equals 5 so as to not split
        print(line)

if __name__ == '__main__':
    read_chinese()