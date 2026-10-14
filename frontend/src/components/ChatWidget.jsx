import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send, Bot, User, Loader2 } from 'lucide-react';

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Witaj! W czym mogę Ci dzisiaj pomóc?' }
  ]);
  const [input, setInput] = useState('');
  
  // Dwa osobne stany do kontroli UI:
  const [isThinking, setIsThinking] = useState(false); // Pokazuje skaczące kropki
  const [isGenerating, setIsGenerating] = useState(false); // Blokuje input na czas całej odpowiedzi

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking]);

  const handleSendMessage = async (e) => {
    if (e && e.preventDefault) {
      e.preventDefault();
    }

    // Blokujemy wysyłanie, gdy trwa proces myślenia LUB pisania
    if (!input.trim() || isGenerating) {
      return;
    }

    const userMessageText = input.trim();
    setInput('');
    
    // Włączamy blokady
    setIsGenerating(true); 
    setIsThinking(true);

    const updatedHistory = [...messages, { role: 'user', content: userMessageText }];
    setMessages(updatedHistory);

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessageText,
          history: messages,
        }),
      });

      if (!response.ok) {
        throw new Error(`Błąd serwera: ${response.status}`);
      }

      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let isFirstChunk = true;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        if (chunk) {
          if (isFirstChunk) {
            isFirstChunk = false;
            setIsThinking(false); // Wyłączamy TYLKO skaczące kropki, ale input DALEJ JEST ZABLOKOWANY!
            
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: chunk }
            ]);
          } else {
            setMessages((prevMessages) => {
              const updated = [...prevMessages];
              const lastIndex = updated.length - 1;
              updated[lastIndex] = {
                ...updated[lastIndex],
                content: updated[lastIndex].content + chunk,
              };
              return updated;
            });
          }
        }
      }
    } catch (error) {
      console.error('Błąd podczas streamingu:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Wystąpił błąd podczas pobierania odpowiedzi.' }
      ]);
    } finally {
      // Dopiero GDY CAŁY STRUMIEŃ SIĘ SKOŃCZY, odblokowujemy input!
      setIsThinking(false);
      setIsGenerating(false);
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
                <div className={`p-3 rounded-2xl text-xs leading-relaxed whitespace-pre-wrap ${
                  msg.role === 'user' 
                    ? 'bg-indigo-500 text-white rounded-tr-none' 
                    : 'bg-white text-slate-800 border border-slate-300 shadow-sm rounded-tl-none'
                }`}>
                  {msg.content}
                </div>
              </div>
            ))}

            {/* LOADER (KROPKI) – widoczny tylko gdy backend przetwarza (isThinking) */}
            {isThinking && (
              <div className="flex gap-1.5 mr-auto max-w-[80%] items-center">
                <div className="w-7 h-7 rounded-full bg-slate-200 text-slate-800 flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="p-3 bg-white border border-slate-300 shadow-sm rounded-2xl rounded-tl-none flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                  <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></span>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* POLE INPUT */}
          <form onSubmit={handleSendMessage} className="p-3 bg-white border-t border-slate-200 flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isGenerating}
              placeholder={isGenerating ? "Asystent odpowiada..." : "Zadaj pytanie..."}
              className="flex-1 px-4 py-3 border border-slate-400 rounded-full text-xs outline-none focus:border-indigo-500 transition-colors disabled:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-400"
            />
            <button
              type="submit"
              disabled={isGenerating || !input.trim()}
              className="w-9 h-9 bg-indigo-500 hover:bg-indigo-700 text-white rounded-full flex items-center justify-center transition-all disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
            >
              {isGenerating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </form>
        </div>
      )}
    </div>
  );
}