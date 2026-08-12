import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send, Bot, User, Loader2 } from 'lucide-react';

const API_URL = "http://127.0.0.1:8000";

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Witaj! W czym mogę Ci dzisiaj pomóc?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userText = input.trim();
    setInput('');

    // Dodajemy wiadomość użytkownika
    const newMessages = [...messages, { role: 'user', content: userText }];
    setMessages(newMessages);
    setLoading(true);

    try {
      // Przygotowujemy historię dla FastAPI (bez powitania startowego)
      const historyForApi = newMessages.slice(1, -1);

      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userText,
          history: historyForApi
        })
      });

      const data = await res.json();

      if (res.ok) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: 'Przepraszam, wystąpił błąd serwera.' }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Brak połączenia z serwerem API.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* PRZYCISK OTWIERAJĄCY CZAT */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="w-14 h-14 bg-indigo-500 hover:bg-indigo-700 text-white rounded-full flex items-center justify-center shadow-xl transition-all duration-300 transform hover:scale-110"
        >
          <MessageSquare className="w-7 h-7" />
        </button>
      )}

      {/* OKNO CZATU */}
      {isOpen && (
        <div className="w-[330px] h-[460px] bg-white rounded-3xl shadow-2xl flex flex-col border border-slate-300 overflow-hidden">
          {/* NAGŁÓWEK */}
          <div className="bg-indigo-500 p-4 text-white flex justify-between items-center">
            <div className="flex items-center gap-3">
              <Bot className="w-5 h-5" />
              <span className="font-semibold text-sm">Asystent AI</span>
            </div>
            <button 
              onClick={() => setIsOpen(false)}
              className="text-white/80 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* WIADOMOŚCI */}
          <div className="flex-1 p-3 overflow-y-auto space-y-3 bg-slate-50/50">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-1.5 max-w-[80%] ${
                  msg.role === 'user' ? 'ml-auto flex-row-reverse' : 'mr-auto'
                }`}
              >
                <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 ${
                  msg.role === 'user' ? 'bg-indigo-500 text-white' : 'bg-slate-200 text-slate-800'
                }`}>
                  {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>
                <div className={`p-3 rounded-2xl text-xs leading-relaxed ${
                  msg.role === 'user' 
                    ? 'bg-indigo-500 text-white rounded-tr-none' 
                    : 'bg-white text-slate-800 border border-slate-300 shadow-sm rounded-tl-none'
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex items-center gap-2 text-slate-600 text-xs italic">
                <Loader2 className="w-4 h-4 animate-spin text-blue-600" /> Asystent generuje odpowiedź...
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* POLE INPUT */}
          <div className="p-3 bg-white border-t border-slate-200 flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Zadaj pytanie..."
              className="flex-1 px-4 py-3 border border-slate-400 rounded-full text-xs outline-none focus:border-indigo-300 transition-colors"
            />
            <button
              onClick={handleSend}
              disabled={loading}
              className="w-9 h-9 bg-indigo-500 hover:bg-indigo-700 text-white rounded-full flex items-center justify-center transition-all disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}