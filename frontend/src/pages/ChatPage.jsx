import React from 'react';
import ConversationHistory from '../components/chat/ConversationHistory';
import ChatWindow from '../components/chat/ChatWindow';

const ChatPage = () => {
  return (
    <div className="flex h-full w-full overflow-hidden">
      <ConversationHistory />
      <ChatWindow />
    </div>
  );
};

export default ChatPage;
