import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Copy, Check, Clock, AlertTriangle } from 'lucide-react';
import ContextPanel from './ContextPanel';

const MessageBubble = ({ message }) => {
  const isUser = message.role === 'user';
  const isError = message.isError;
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (isUser) {
    return (
      <div className="flex justify-end mb-6">
        <div className="bg-primary text-primary-foreground max-w-[80%] rounded-2xl rounded-tr-sm px-5 py-3 shadow-sm">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex justify-start mb-6">
        <div className="bg-destructive/10 border border-destructive/20 text-destructive max-w-[80%] rounded-2xl rounded-tl-sm px-5 py-3 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <p className="whitespace-pre-wrap text-sm">{message.content}</p>
        </div>
      </div>
    );
  }

  // AI Response
  return (
    <div className="flex justify-start mb-6 group">
      <div className="bg-card border border-border max-w-[90%] rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm w-full">
        
        {/* Markdown Content */}
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Metadata Footer */}
        <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            {message.inferenceTime && (
              <span className="flex items-center gap-1" title="Inference Time">
                <Clock className="w-3.5 h-3.5" />
                {message.inferenceTime.toFixed(2)}s
              </span>
            )}
            
            {message.contextUsed && message.contextUsed.length > 0 && (
              <ContextPanel context={message.contextUsed} />
            )}
          </div>

          <button
            onClick={handleCopy}
            className="flex items-center gap-1 hover:text-foreground transition-colors p-1"
            title="Copy answer"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>

      </div>
    </div>
  );
};

export default MessageBubble;
