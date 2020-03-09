import Socket from "../util/Socket";
import { testEPs } from "../configs/Config.json";

const { testReviewEP } = testEPs;

async function testReview(reviewToTest) {

  const payLoad = {
    review: reviewToTest
  };

  return await Socket.POST(testReviewEP, payLoad);
}

export default {
    testReview
};
