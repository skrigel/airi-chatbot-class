import { useState, useRef } from "react";
import { BotIcon, XIcon } from "lucide-react";
import { Header } from "../../components/header";
import { Chat } from "../chat/chat";
import { message } from "../../interfaces/interfaces";
import { v4 as uuidv4 } from "uuid";

const API_URL = "";

const WELCOME: message = {
  content: "Hi! I'm your AI assistant to help you navigate the AI Risk repository. How can I help you today?",
  role: "assistant",
  id: uuidv4(),
};

export function Home() {
  const [isOpen, setOpen] = useState(false);
  const [previousMessages, setPreviousMessages] = useState<message[]>([]);
  const [currentMessage, setCurrentMessage] = useState<message | null>(WELCOME);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messageHandlerRef = useRef<((event: MessageEvent) => void) | null>(null);

  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  const cleanupMessageHandler = () => {
    if (messageHandlerRef.current) {
      messageHandlerRef.current = null;
    }
  };

  async function handleSubmit(text?: string) {
    if (isLoading) return;
    const messageText = text;
    if (!messageText || !messageText.trim()) return;

    const userMessage: message = { content: messageText, role: "user", id: uuidv4() };
    setPreviousMessages((prev) => [...prev, userMessage]);

    const loadingMessage: message = { content: "Loading...", role: "assistant", id: "loading" };
    setCurrentMessage(loadingMessage);

    try {
      const stream = await fetch(`${API_URL}api/v1/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messageText }),
      });

      if (!stream.body) throw new Error("Failed!!");

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
              if (parsed.related_documents) {
                // In the home page, we don't have a place to display related documents yet.
                // We can just ignore them for now.
              } else {
                accumulatedText += parsed;
                const currMess: message = {
                  content: accumulatedText,
                  role: "assistant",
                  id: uuidv4(),
                };
                setCurrentMessage(currMess);
              }
            } catch (err) {
              console.error("Error parsing line:", line, err);
            }
          }
        }
      }

      const botMessage: message = {
        content: accumulatedText,
        role: "assistant",
        id: uuidv4(),
      };

      setPreviousMessages((prev) => [...prev, botMessage]);
      setCurrentMessage(null);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: message = {
        content: "Error talking to server.",
        role: "assistant",
        id: uuidv4(),
      };
      setPreviousMessages((prev) => [...prev, errorMessage]);
      setCurrentMessage(null);
    } finally {
      setIsLoading(false);
      cleanupMessageHandler();
    }
  }

  return (
    <div className="relative flex flex-col min-h-screen bg-white text-black">
      <Header />

      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center text-center px-6 py-16">
        <div className="max-w-3xl space-y-6">
          {/* Title + Bot Icon */}
          <div className="flex justify-center items-center">
            <div className="w-20 h-20 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center">
              <BotIcon size={40} />
            </div>
            <h1 className="text-4xl sm:text-5xl font-bold text-gray-900">
              Welcome to the AI Risk Repository
            </h1>
          </div>

          <p className="text-lg text-gray-600">
            I’m your helpful companion for researching AI risks, answering questions, and guiding your exploration.
          </p>

          {/* Buttons */}
          <div className="flex flex-col sm:flex-row justify-center gap-4 pt-4">
            <button
              onClick={handleOpen}
              className="px-6 py-3 bg-red-500 text-white rounded-full hover:bg-red-600 transition shadow-md"
            >
              Open Chat
            </button>
      
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="text-center py-4 text-gray-500 border-t border-gray-200">
        &copy; {new Date().getFullYear()} AI Risk Repository. All rights reserved.
      </footer>

      {/* Chat Popup */}
      <div
        className={`fixed bottom-5 right-5 z-50 transition-all duration-300 ease-in-out ${
          isOpen
            ? "translate-y-0 opacity-100 scale-100"
            : "translate-y-5 opacity-0 scale-95 pointer-events-none"
        }`}
        style={{
          width: "90vw",
          maxWidth: "400px",
          height: "65vh",
        }}
      >
        <div className="w-full h-full bg-white shadow-xl rounded-2xl overflow-hidden flex flex-col border border-gray-200">
          {/* Close Button */}
          <div className="flex justify-end p-2 border-b border-gray-200 bg-gray-100">
            <button
              className="text-gray-600 hover:text-red-500"
              onClick={handleClose}
            >
              <XIcon size={20} />
            </button>
          </div>

          {/* Chat Area */}
          <div className="flex-1 overflow-hidden bg-gray-50">
            <Chat
              previousMessages={previousMessages}
              currentMessage={currentMessage}
              handleSubmit={handleSubmit}
              isLoading={isLoading}
            />
          </div>
        </div>
      </div>

      {/* Floating Chat Icon */}
      {!isOpen && (
        <button
          onClick={handleOpen}
          className="fixed bottom-5 right-5 bg-red-500 text-white p-4 rounded-full shadow-lg hover:bg-red-600 transition z-40"
        >
          <BotIcon size={28} />
        </button>
      )}
    </div>
  );
}
