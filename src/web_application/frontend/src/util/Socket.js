import Axios from "axios";

import Config from "../configs/Config.json";

const { baseUrl, pollLimit, gatewayEPs } = Config;
const localStorage = require("local-storage");

const HTTPMethod = Object.freeze({
  GET: "GET",
  POST: "POST",
  DELETE: "DELETE"
});

function initSocket() {
  const { common } = Axios.defaults.headers;

  Axios.defaults.baseURL = baseUrl;
}

async function GET(path, params) {
  return await sendHTTP(HTTPMethod.GET, path, params, null);
}

async function POST(path, data, headers, service) {
  return await sendHTTP(HTTPMethod.POST, path, data, headers, service);
}

async function sendHTTP(method, path, data, headers, service) {

  switch (method) {
    case HTTPMethod.GET:
      console.log("Inside GET Case");
      break;

    case HTTPMethod.POST:
      console.log("Inside a POST Case");
      return await Axios.post(path, data);
      break;

    default:
    // Should never reach here
  }
}

export default {
  initSocket,
  GET,
  POST,
};
