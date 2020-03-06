from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return "This is the server for CS 175 Detecting Fake Reviews"
    # return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
