import React, { Component } from "react";

import "./css/nav-bar.css";

class NavBar extends Component {

    render() {

        return (
            <nav className="nav-bar">
                <p className="nav-link">CS 175: Detecting Fake Reviews</p>
            </nav>
        );
    }
}

export default NavBar;
