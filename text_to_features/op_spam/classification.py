# Functions to create various text classification models
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
import functions

def logisticRegression(X, Y, test_fraction=0.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    classifier = linear_model.LogisticRegression(penalty='l2', fit_intercept=True)
    classifier.fit(X_train, Y_train)
    test_accuracy = classifier.score(X_test, Y_test)
    print(' accuracy:', format(100 * test_accuracy, '.2f'))

def naiveBayes(X, Y, test_fraction=0.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    mnb = MultinomialNB()
    mnb.fit(X_train, Y_train)
    test_accuracy = mnb.score(X_test, Y_test)
    print(' accuracy:', format(100 * test_accuracy, '.2f'))

if __name__ == '__main__':
    print('Running Program')
    # get BOW and ground truth from functions.py
    X, Y = functions.create_reviews_scores_arrays()
    logisticRegression(X,Y)
