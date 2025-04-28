import { ChatInput } from "../../components/chatinput";
import { useScrollToBottom } from '../../components/use-scroll-to-bottom';
import { useState, useRef } from "react";
import { message } from "../../interfaces/interfaces";
import {v4 as uuidv4} from 'uuid';


const API_URL = 'http://localhost:5000/';

export function Chat() {
  const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
  const [previousMessages, setPreviousMessages] = useState<message[]>([]);
  const [currentMessage, setCurrentMessage] = useState<message | null>(null);
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
    if (!messageText.trim()) return;
  
    setIsLoading(true);
    setQuestion("");
  
    const userMessage: message = { content: messageText, role: "user", id: uuidv4() };
  
    // 1. Immediately add user message
    setPreviousMessages(prev => [...prev, userMessage]);
  
    // 2. Add a temporary bot loading message
    const loadingMessage: message = { content: "Loading...", role: "assistant", id: "loading" };
    setCurrentMessage(loadingMessage);
  
    try {
      const response = await fetch(`${API_URL}api/v1/sendMessage`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: messageText })
      });
  
      const data = await response.json();
      console.log('Bot response data:', data);
  
      if (data.error) {
        throw new Error(data.error);
      }
  
      const botMessage: message = {
        content: data.response,
        role: "assistant",
        id: uuidv4()
      };
  
      // 3. Move both loading message and real bot message into previousMessages
      setPreviousMessages(prev => [...prev, botMessage]);
      setCurrentMessage(null);
  
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: message = {
        content: "Error talking to server.",
        role: "assistant",
        id: uuidv4()
      };
  
      setPreviousMessages(prev => [...prev, errorMessage]);
      setCurrentMessage(null);
    } finally {
      setIsLoading(false);
      cleanupMessageHandler();
    }
  }
  
  return (
    <div className="relative h-full w-full bg-white rounded-xl flex flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-2" ref={messagesContainerRef}>
  {previousMessages.map((msg) => (
    <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
      <div className={`p-2 rounded-lg max-w-xl ${
        msg.role === "user" ? "bg-blue-200" : "bg-gray-200"
      }`}>
        {msg.content}
      </div>
    </div>
  ))}

  {currentMessage && (
    <div key={currentMessage.id} className="flex justify-start">
      <div className="p-2 rounded-lg max-w-xl bg-gray-200">
        {currentMessage.content}
      </div>
    </div>
  )}

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


