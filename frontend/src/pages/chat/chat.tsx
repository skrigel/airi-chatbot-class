import { ChatInput } from "../../components/chatinput";
import { useScrollToBottom } from '../../components/use-scroll-to-bottom';
import { useState, useRef } from "react";
import { message } from "../../interfaces/interfaces";
import {v4 as uuidv4} from 'uuid';

export function Chat() {
  const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messageHandlerRef = useRef<((event: MessageEvent) => void) | null>(null);

  const cleanupMessageHandler = () => {
    if (messageHandlerRef.current) {
      messageHandlerRef.current = null;
    }
  };

  async function handleSubmit(text?: string) {
    if (isLoading) return;

    const messageText = text || question;
    setIsLoading(true);
    cleanupMessageHandler();

    const traceId = uuidv4();
    setMessages(prev => [...prev, { content: messageText, role: "user", id: traceId }]);
    setQuestion("");

    try {
      const messageHandler = (event: MessageEvent) => {
        setIsLoading(false);
        if (event.data.includes("[END]")) return;

        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          const newContent = lastMessage?.role === "assistant"
            ? lastMessage.content + event.data
            : event.data;

          const newMessage = { content: newContent, role: "assistant", id: traceId };
          return lastMessage?.role === "assistant"
            ? [...prev.slice(0, -1), newMessage]
            : [...prev, newMessage];
        });

        if (event.data.includes("[END]")) {
          cleanupMessageHandler();
        }
      };

      messageHandlerRef.current = messageHandler;
    } catch (error) {
      console.error("WebSocket error:", error);
      setIsLoading(false);
    }
  }

  return (
    <div className="relative h-full w-full bg-white rounded-xl flex flex-col">
      {/* Messages */}
      <div
        className="flex-1 overflow-y-auto px-4 py-2 space-y-2"
        ref={messagesContainerRef}
      >
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`p-2 rounded-lg max-w-xl ${
              msg.role === "user" ? "bg-blue-200 self-end" : "bg-gray-200 self-start"
            }`}
          >
            {msg.content}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <div className="p-4 border-t">
        <ChatInput
          question={question}
          setQuestion={setQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}


