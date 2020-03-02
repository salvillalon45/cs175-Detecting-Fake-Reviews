import Axios from "axios";

import MoviesConfig from "../../configs/MoviesConfig.json";

const { baseUrl } = MoviesConfig;

const HTTPMethod = Object.freeze({
  GET: "GET",
  POST: "POST",
  DELETE: "DELETE"
});

function initSocket() {
  console.log("Inside initSocket() of MoviesSocket");
  console.log("What is baseURL:: ", baseUrl);
}

async function GET(path, params) {
  return await sendHTTP(HTTPMethod.GET, path, params);
}

async function POST(path, data) {
  return await sendHTTP(HTTPMethod.POST, path, data);
}

async function DELETE(path) {
  return await sendHTTP(HTTPMethod.DELETE, path);
}

async function sendHTTP(method, path, data) {
  switch (method) {
    case HTTPMethod.GET:

      let searchQuery1 = "";
      let searchQuery2 = "";
      let searchQuery3 = "";
      let email = data.headers.email;
      let session_id = data.headers.session_id;

      if (data.filterType === "title") {
        searchQuery1 = data.searchQuery;
        searchQuery2 = null;
        searchQuery3 = null;
      }
      else if (data.filterType === "director") {
        searchQuery1 = null;
        searchQuery2 = data.searchQuery;
        searchQuery3 = null;
      }
      else if (data.filterType === "year") {
        searchQuery1 = null;
        searchQuery2 = null;
        searchQuery3 = data.searchQuery;
      }

      return await Axios.get(path, {
        headers: {
          email: email,
          session_id: session_id,
          transaction_id: 1
        },
        params : {
          title: searchQuery1,
          director: searchQuery2,
          year: searchQuery3
        }
      });
    case HTTPMethod.POST:
      return await Axios.post(path, data);
    case HTTPMethod.DELETE:
      return await Axios.delete(path);
    default:
      throw new Error("Invalid HTTPMethod Given");
  }
}

export default {
  initSocket,
  GET,
  POST,
  DELETE
};
