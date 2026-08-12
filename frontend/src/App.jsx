import React from 'react';
import AdminPanel from './components/AdminPanel';
import ChatWidget from './components/ChatWidget';

function App() {
  return (
    <div className="min-h-screen bg-slate-100 font-sans antialiased p-4">
      {/* Glówny widok: Panel Admina */}
      <AdminPanel />

      {/* Pływający widżet czatu w prawym dolnym rogu */}
      <ChatWidget />
    </div>
  );
}

export default App;