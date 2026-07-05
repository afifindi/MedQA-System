import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

const ContextPanel = ({ context }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1 hover:text-foreground transition-colors rounded bg-secondary/50 px-2 py-1"
      >
        <BookOpen className="w-3.5 h-3.5" />
        <span>{context.length} Sources</span>
        {isOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      </button>

      {isOpen && (
        <div className="absolute left-0 top-full mt-2 w-80 max-h-60 overflow-y-auto bg-popover text-popover-foreground border border-border shadow-lg rounded-md p-3 z-20 text-sm">
          <h4 className="font-semibold mb-2 text-xs uppercase tracking-wider text-muted-foreground border-b border-border pb-1">
            Retrieved Context
          </h4>
          <div className="space-y-3 mt-2">
            {context.map((doc, idx) => (
              <div key={idx} className="bg-muted/30 p-2 rounded text-xs border border-border/50">
                <div className="flex justify-between items-start mb-1">
                  <span className="font-medium text-primary">Source {idx + 1}</span>
                  {doc.score !== undefined && (
                    <span className="text-[10px] bg-secondary px-1.5 py-0.5 rounded text-muted-foreground">
                      Score: {doc.score.toFixed(3)}
                    </span>
                  )}
                </div>
                <p className="line-clamp-4 text-muted-foreground leading-relaxed">
                  {doc.content || doc.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ContextPanel;
