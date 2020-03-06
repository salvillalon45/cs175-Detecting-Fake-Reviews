import React, { Component } from "react";

import Content from "./Content";
import NavBar from "./NavBar";
import Axios from "axios";

/*
  Remember when passing around variable and functions

  If you want componenets down the tree to VIEW your 
  variable, simply pass that varaible

  If you want componenets down the tree to MODIFY your
  varaible, then pass a function down the tree that allows
  the components to modify it as needed

  The function should be made in the Component that "owns" that 
  varaible, as in which Component has it as part of its state
  because only that component can modify its own state.
  Pass variables and functions into Components like so:

    <Componenet var={value} func={value}/>

  IMPORTANT: DO NOT CALL THE FUNCTION

  func={function}   WORKS
  func={function()} DOES NOT WORK, This is calling the function 
                                   and storing the return value

  To get the passed variables and functions from inside 
  the Componenet call `this.props.<NAME>`
*/

const localStorage = require("local-storage");

class App extends Component {
    state = {
        loggedIn: this.checkedLoggedIn()
    };

    handleLogin = (email, session_id) => {
        console.log("Inside handleLogin()");

        const { common } = Axios.defaults.headers;
        localStorage.set("email", email);
        localStorage.set("session_id", session_id);

        common["email"] = email;
        common["session_id"] = session_id;

        this.setState({
            loggedIn: true
        });

        if (this.state.loggedIn) {
            return true;
        }
        else {
            return false;
        }
    };

    checkedLoggedIn() {
        console.log("Inside checkedLogin()");

        return (
            localStorage.get("email") !== undefined &&
            localStorage.get("session_id") !== undefined &&
            localStorage.get("email") !== null &&
            localStorage.get("session_id") !== null
        );
    }


  render() {

      const { loggedIn } = this.state;
      console.log("Inside App Component");

      return (
          <div className="app">
              <NavBar loggedIn={loggedIn} />
              <Content loggedIn={loggedIn} handleLogin={this.handleLogin}/>
          </div>
      );
  }
}

export default App;
