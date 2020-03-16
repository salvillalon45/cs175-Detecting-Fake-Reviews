import os, sys
sys.path.append('../op_spam')
sys.path.append('../yelp')
import functions
import classification
import yelp_parser
import json
import ijson

import numpy as np

def getModel():
    print('getting model')
    reviews, scores = yelp_parser.get_chi_hotel_review_score_list()
    reviews, vec = functions.create_bow_from_reviews(reviews)
    print('first reviews')
    print(reviews.shape)
    logistic = classification.logistic_regression(reviews, scores)
    return logistic, vec

def getUnlabeledYelp():
    print('getting unlabeled yelp')
    filename = '../../datasets/unlabeledYelp/review.json'

    count = 0
    texts = []
    for line in open(filename):
        line = line.strip()
        # print(line)
        data = json.loads(line)
        text = data['text']
        text = text.strip()
        # print(text)
        texts.append(text)
        count = count + 1
        # print(count)
        if count > 1000000:
            break
    return texts

def create_bow(reviews, vec):
    reviews = vec.transform(reviews)
    return reviews

if __name__ == '__main__':
    print('Results on unlabeled Yelp data')

    #look at yelp data set

    #create model on yelp reviews
    logistic, vec = getModel()

    #get reviews from yelp file
    old_reviews = getUnlabeledYelp()

    # reviews, vec = functions.create_bow_from_reviews(reviews)
    reviews = create_bow(old_reviews, vec)

    print('second reviews')
    print(reviews.shape)

    predictions = logistic.predict(reviews)

    print(predictions)

    print(len(predictions))
    print(sum(predictions))

    true = float(sum(predictions))
    total = float(len(predictions))
    percent_true = true/total
    print(percent_true)

    class_probabilities = logistic.predict_proba(reviews)
    class_probabilities = class_probabilities[:,1]
    print(class_probabilities)

    arr = np.array(class_probabilities)
    top = arr.argsort()[-3:][::-1]
    bottom = arr.argsort()[0:3]

    print('top: ', top)
    print(arr[top[0]], arr[top[1]], arr[top[2]])

    print('bottom: ', bottom)
    print(arr[bottom[0]], arr[bottom[1]], arr[bottom[2]])

    print('most truthful')
    print(old_reviews[top[0]])
    print('NEXT',old_reviews[top[1]])
    print('NEXT',old_reviews[top[2]])

    print('most deceptive')
    print(old_reviews[bottom[0]])
    print('NEXT',old_reviews[bottom[1]])
    print('NEXT',old_reviews[bottom[2]])


    #find percentage of true vs false reviews

    #find some especially true and especially false reviews