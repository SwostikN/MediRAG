
  import { createRoot } from "react-dom/client";
  import App from "./app/App.tsx";
  import { SettingsProvider } from "./lib/settings";
  import "./styles/index.css";

  createRoot(document.getElementById("root")!).render(
    <SettingsProvider>
      <App />
    </SettingsProvider>,
  );
