import React from "react";
import ReactDOM from "react-dom/client";
import { Toaster } from "react-hot-toast";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
    <Toaster
      position="bottom-right"
      toastOptions={{
        style: {
          background: "rgba(23,27,40,0.92)",
          color: "#e6ebf2",
          border: "1px solid rgba(255,255,255,0.08)",
          backdropFilter: "blur(8px)",
        },
      }}
    />
  </React.StrictMode>,
);
