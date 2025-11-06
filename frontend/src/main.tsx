import React from 'react';
import ReactDOM from 'react-dom/client';
// --- Use Relative Paths ---
import App from './App';         // App.tsx is in the same directory (src)
import './index.css';       // index.css is in the same directory (src)
import { AuthProvider } from './AuthContext'; // AuthContext.tsx is in the same directory (src)
// --- End Relative Paths ---

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>,
);

