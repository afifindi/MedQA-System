import React from 'react';
import { useChat } from '../../context/ChatContext';
import { MessageSquare, Trash2 } from 'lucide-react';

const ConversationHistory = () => {
  const { messages, clearChat } = useChat();

  // Group messages to show only user questions in history
  const questions = messages.filter((m) => m.role === 'user');

  return (
    <div className="w-64 border-l border-border bg-card flex flex-col h-full hidden lg:flex">
      <div className="p-4 border-b border-border flex justify-between items-center">
        <h2 className="font-medium text-sm text-muted-foreground uppercase tracking-wider">
          History
        </h2>
        <button
          onClick={clearChat}
          className="text-muted-foreground hover:text-destructive p-1 rounded-md transition-colors"
          title="Clear Conversation"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2">
        {questions.length === 0 ? (
          <div className="text-center text-sm text-muted-foreground p-4 mt-10">
            No active conversation.
          </div>
        ) : (
          <ul className="space-y-1">
            {questions.map((q) => (
              <li key={q.id}>
                <button
                  className="w-full text-left p-2 text-sm rounded-md hover:bg-secondary truncate text-foreground flex items-center gap-2 transition-colors"
                  title={q.content}
                >
                  <MessageSquare className="w-4 h-4 flex-shrink-0 text-muted-foreground" />
                  <span className="truncate">{q.content}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default ConversationHistory;
