import { ChatInput } from "../../components/chatinput";
import { useScrollToBottom } from '../../components/use-scroll-to-bottom';
import { useState, useRef } from "react";
import { message } from "../../interfaces/interfaces";
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from "react-markdown";

const API_URL = '';

const QUESTIONS = ["Find AI risk papers related to “Pre-deployment” timing",
       "What are the main risk categories in the AI Risk Database v3?"];

const WELCOME: message = { content: "Hi! I'm your AI assistant to help you navigate the AI Risk repository. How can I help you today?", role: "assistant", id: uuidv4() };

export function Chat() {


  const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
  const [previousMessages, setPreviousMessages] = useState<message[]>([]);
  const [currentMessage, setCurrentMessage] = useState<message | null>(WELCOME);
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
    setQuestion("");

    const userMessage: message = { content: messageText, role: "user", id: uuidv4() };
    setPreviousMessages(prev => [...prev, userMessage]);

    const loadingMessage: message = { content: "Loading...", role: "assistant", id: "loading" };
    setCurrentMessage(loadingMessage);

    try {
      const stream = await fetch(`${API_URL}api/v1/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messageText })
      });

      if (!stream.body) throw new Error('Failed!!');

      const reader = stream.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = "";
      let accumulatedText = "";

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        buffer += decoder.decode(value, { stream: !done });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.trim()) {
            try {
              const parsed = JSON.parse(line);
              accumulatedText += parsed;
              const currMess: message = { content: accumulatedText, role: 'assistant', id: uuidv4() };
              setCurrentMessage(currMess);
            } catch (err) {
              console.error("Error parsing line:", line, err);
            }
          }
        }
      }

      const botMessage: message = {
        content: accumulatedText,
        role: "assistant",
        id: uuidv4()
      };

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

      {/* Airtable-style header */}
      <div className="p-6 border-b border-gray-200 text-center">
        <div className="flex justify-center items-center space-x-2 mb-2">
          <div className="w-2.5 h-2.5 bg-red-400 rounded-full animate-ping" />
          <div className="text-gray-800 font-medium text-lg">How can I help?</div>
        </div>
        <div className="flex flex-wrap justify-center gap-2 text-sm text-gray-600">
          {QUESTIONS.map((question, index) => (
            <button key={index} className="bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-full">
              {question}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3" ref={messagesContainerRef}>
        {previousMessages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`px-4 py-2 rounded-2xl max-w-xl whitespace-pre-wrap shadow-sm text-sm ${
              msg.role === "user" ? "bg-blue-100 text-black" : "bg-gray-100 text-black"
            }`}>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}

        {currentMessage && (
          <div key={currentMessage.id} className="flex justify-start">
            <div className="px-4 py-2 rounded-2xl max-w-xl bg-gray-100 shadow-sm text-sm text-black">
              <ReactMarkdown>{currentMessage.content}</ReactMarkdown>
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
