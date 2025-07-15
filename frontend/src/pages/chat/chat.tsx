import { ChatInput } from "../../components/chatinput";
import { useScrollToBottom } from '../../components/use-scroll-to-bottom';
import { useState } from "react";
import { message } from "../../interfaces/interfaces";
import ReactMarkdown from "react-markdown";

const QUESTIONS = ["Find AI risk papers related to “Pre-deployment” timing",
       "What are the main risk categories in the AI Risk Database v3?"];

interface ChatProps {
  previousMessages: message[];
  currentMessage: message | null;
  handleSubmit: (text?: string) => Promise<void>;
  isLoading: boolean;
}

export function Chat({ previousMessages, currentMessage, handleSubmit, isLoading }: ChatProps) {
  const [messagesContainerRef, messagesEndRef] = useScrollToBottom<HTMLDivElement>();
  const [question, setQuestion] = useState<string>("");

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
            <button
              key={index}
              className="bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-full"
              onClick={() => handleSubmit(question)}
            >
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
