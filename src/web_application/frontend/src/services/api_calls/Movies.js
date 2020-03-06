import MoviesSocket from "../sockets/MoviesSocket";
import { moviesEPs } from "../../configs/MoviesConfig.json";

const { searchEP } = moviesEPs;
const baseURL = "http://localhost:1795/api/";

async function search(filterType, searchQuery, headers) {

    const params = {
        filterType: filterType,
        searchQuery: searchQuery,
        headers: headers
    };

    let path = baseURL + searchEP;

    return await MoviesSocket.GET(path, params);
}

export default {
    search
};
