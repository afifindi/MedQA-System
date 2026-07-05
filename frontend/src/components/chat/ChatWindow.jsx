import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '../../context/ChatContext';
import MessageBubble from './MessageBubble';
import LoadingSpinner from '../common/LoadingSpinner';
import { Send, Stethoscope } from 'lucide-react';

const ChatWindow = () => {
  const { messages, isLoading, error, sendMessage } = useChat();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    sendMessage(input);
    setInput('');
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-background relative">
      {/* Header */}
      <header className="p-4 border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-10">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Stethoscope className="w-5 h-5 text-primary" />
          Medical Assistant
        </h2>
      </header>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground max-w-md mx-auto text-center space-y-4">
            <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4">
              <Stethoscope className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-xl font-medium text-foreground">How can I help you today?</h3>
            <p className="text-sm">
              I am a medical AI assistant using Retrieval-Augmented Generation to provide answers grounded in trusted medical literature.
            </p>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            
            {isLoading && (
              <div className="flex items-center gap-3 text-muted-foreground p-4 bg-muted/30 rounded-lg max-w-[80%]">
                <LoadingSpinner size="sm" />
                <span className="text-sm font-medium animate-pulse">Analysing medical knowledge base...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 bg-card border-t border-border">
        <div className="max-w-3xl mx-auto relative">
          <form onSubmit={handleSubmit} className="relative flex items-center">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a medical question..."
              className="w-full bg-background border border-border rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none max-h-32 min-h-[52px]"
              rows="1"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="absolute right-2 p-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-opacity hover:bg-primary/90"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="text-xs text-center text-muted-foreground mt-2">
            AI can make mistakes. Always consult a healthcare professional.
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatWindow;
