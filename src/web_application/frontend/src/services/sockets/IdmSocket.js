import Axios from "axios";

import IdmConfig from "../../configs/IdmConfig.json";

const { baseUrl } = IdmConfig;

const HTTPMethod = Object.freeze({
  GET: "GET",
  POST: "POST",
  DELETE: "DELETE"
});

function initSocket() {
  console.log("Inside initSocket() of IdmSocket");
  console.log("What is baseURL:: ", baseUrl);
}

async function GET(path) {
  return await sendHTTP(HTTPMethod.GET, path);
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
      return await Axios.get(path);
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
