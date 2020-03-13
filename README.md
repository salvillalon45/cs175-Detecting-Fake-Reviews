# Team SBA: Salvador Villalon, Brandon Teran, Andrew Michael
# Project: cs175-Detecting-Fake-Reviews




## Files
- src/main.py
    - This script is meant to bring together all of the project's functionality. Here we parse the datasets, train the Bag of Words and Doc2Vec models, and train and predict using our classifiers.
- src/functions.py
    - This file has the utility functions needed to add more features and parse. It also contains functions that call the .train() and .predict() methods on each of our classifiers.
- src/embeddings.py
    - This script provides utility functions to create a corpus from the reviews in the form of [TaggedDocument]. It also provides functions to create, train and get a Doc2Vec embedding of said corpus.
- src/classification.py
    - This script provides helper functions to create each of the following classifiers:
        - Logistic Regression Classifier
        - Naïve Bayes Classifier (not used)
        - K Nearest Neighbors Classifier
        - Decision Tree Classifier
        - Random Forest Classifier
- src/yelp_parser.py
    - This script parses the Yelp Review Dataset. It offers an functions that allow to retrieve the reviews, scores and length of each review in array format.
- src/web_application
    - Contains a simple web application, running on React and Flask
- src/resources
    - Contains Jupyter Notebook used to test Doc2Vec model, assignment 1 reference, and the reviews we used for our Google Forms survey.


## Resources
- [An Introduction to Bag-of-Words in NLP](https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428)
- [Ultimate Guide to Understand and Implement Natural Language Processing (with codes in Python)](https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/)
- [Preparing Your Dataset for Machine Learning: 8 Basic Techniques That Make Your Data Better](https://www.altexsoft.com/blog/datascience/preparing-your-dataset-for-machine-learning-8-basic-techniques-that-make-your-data-better/)
- [Introduction to Classification Algorithms](https://dzone.com/articles/introduction-to-classification-algorithms)
