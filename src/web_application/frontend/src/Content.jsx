import React, { Component } from "react";

import Home from "./pages/Home";
import "./css/index.css";

class Content extends Component {

    render() {

        return (
            <div className="content">
                <Home />
            </div>
        );
    }
}

export default Content;
