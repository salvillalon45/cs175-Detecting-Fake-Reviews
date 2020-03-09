import IdmSocket from "../sockets/IdmSocket";
import { idmEPs } from "../../configs/IdmConfig.json";

const { loginEP, registerEP } = idmEPs;

const baseURL = "http://localhost:1796/api/";

async function login(email, password) {
  const payLoad = {
    email: email,
    password: password.split("")
  };

  let path = baseURL + loginEP;

  return await IdmSocket.POST(path, payLoad);
}

async function register(email, password) {
  const payLoad = {
    email: email,
    password: password.split("")
  };

  let path = baseURL + registerEP;

  return await IdmSocket.POST(path, payLoad);
}

export default {
  login,
  register
};
