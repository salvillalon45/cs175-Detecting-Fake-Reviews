from flask import Flask
from flask import request
import functions
import nltk

app = Flask(__name__)

reviews, scores, length_of_reviews = functions.parse_opspam_reviews()
# reviews, scores = functions.parse_yelp_reviews()
model, train_targets, train_regressors, test_targets, test_regressors, train_x = functions.train_model_from_corpus(reviews, scores)


@app.route("/testReview", methods=["POST"])
def test_review_endpoint():
    print("Inside test_review_endpoint()")

    review = request.get_json()["review"]
    print("User Review:: ", review)

    review_tokens = nltk.word_tokenize(review)
    prediction = functions.get_prediction(review_tokens, model, train_targets, train_regressors, test_targets, test_regressors, train_x)

    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True)
