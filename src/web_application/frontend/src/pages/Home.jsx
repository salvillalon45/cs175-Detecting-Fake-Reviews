import React, { Component } from "react";

import "../css/form.css";
import "../css/Home.css";
import API from "../services/API";

class Home extends Component {
  state = {
      inputReview: "",
      result: ""
  };

  updateInputReview = (event) => {
      const inputReview = event.target.value;

      this.setState({
          "inputReview": inputReview
      });
  };

  updateResult = (result) => {
      this.setState({
         "result":  result
      });
  };

  handleSubmit = (event) => {
      console.log("Inside handleSubmit() of Home.jsx");
      event.preventDefault();

      const review = this.state.inputReview;

      API.testReview(review)
          .then(response =>
              this.updateResult(response.data))
              // alert(JSON.stringify(response.data, null, 4)))

          .catch(
            error => alert(error)
          );
  };

  handleReviewResult = () => {
      if (this.state.result) {
          return <p>This review is real!</p>;
      }
      else {
          return <p>This review is fake!</p>;
      }
  };

  render() {
      const inputReview = this.state.inputReview;

      return (
          <div className="form-box">
              <h1>Test Your Review</h1>

              <form onSubmit={this.handleSubmit}>

                  {/* Review Input Field */}
                  <div className="review-input-field">
                      <p>There are many reviews for many things that are online. There can be reviews for a hotel, a laptop, or a restaurant. Many of these reviews can also be fake, giving the manufacturer or seller higher ratings for products that might not be as beneficial to users they may seem.</p>
                      <p>Our project addresses this problem by attempting to classify reviews as either fake (written by bots or hiring someone to write the reviews) or real (genuine). We plan on using the following types of types of classification algorithms in Machine Learning: Logistic Regression, Naive Bayes Classifier, Nearest Neighbor, Decision Trees, Random Forest. We plan on seeing how our algorithm improves by using word embeddings.</p>
                      <textarea className="form-input" placeholder="Place a review and click 'Test It' to run it through our algorithm" name="message" rows="10" cols="30" value={inputReview} onChange={this.updateInputReview}/>
                  </div>

                  <div className="result-box">
                      <h2>Prediction Result</h2>
                      <p>Here goes the result when you press Test It</p>
                      <div className="result">
                          {
                              this.state.result !== "" && this.handleReviewResult()
                          }
                          {/*{*/}
                          {/*    this.function()*/}
                          {/*      // !this.state.result && this.state.result !== "" && (*/}
                          {/*      //   <p>This review is Fake!</p>*/}
                          {/*      // )*/}
                          {/*}*/}
                          {/*  {*/}
                          {/*      this.state.result && (*/}
                          {/*        <p>This review is real!</p>*/}
                          {/*      )*/}
                          {/*  }*/}
                      </div>
                      {/*{*/}
                      {/*    !this.state.result && this.state.result !== "" && (*/}
                      {/*        <div className="result">*/}
                      {/*            <p>This review is Fake!</p>*/}
                      {/*        </div>*/}
                      {/*    )*/}
                      {/*}*/}
                      {/*{*/}
                      {/*    this.state.result && (*/}
                      {/*        <div className="result">*/}
                      {/*            <p>This review is real!</p>*/}
                      {/*        </div>*/}
                      {/*    )*/}
                      {/*}*/}

                  </div>

                  {/* Button */}
                  <div className="review-input-button">
                      <button className="form-button">Test It</button>
                  </div>

              </form>
          </div>
    );
  }
}

export default Home;
