import { useState } from "react";
import { BotIcon, XIcon } from "lucide-react"; // Just an example, replace with your actual icons
import { Header } from "../../components/header";
import { Chat } from '../chat/chat';

export function Home() {
  const [isOpen, setOpen] = useState(false);

  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  return (
    <div className="relative flex flex-col min-h-screen bg-gray-50 text-black">
      <Header />

      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center text-center px-6">
        <div className="max-w-3xl">
          <h1 className="text-5xl font-bold mb-4 text-gray-900">
            Welcome to the AI Risk Repository
          </h1>
          <p className="text-lg text-gray-600 mb-6">
            Your AI companion for instant help, smart replies, and real-time support.
          </p>
       
          <button
            onClick={handleOpen}
            className="px-6 py-3 bg-red-500 text-white rounded-full hover:bg-red-600 transition shadow-md"
          >
            Open Chat
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="text-center py-4 text-gray-500 border-t border-gray-200">
        &copy; {new Date().getFullYear()} AI Risk Repository. All rights reserved.
      </footer>

      {/* Chat Popup */}
      <div
        className={`fixed bottom-5 right-5 z-50 transition-all duration-300 ease-in-out ${
          isOpen ? "translate-y-0 opacity-100 scale-100" : "translate-y-5 opacity-0 scale-95 pointer-events-none"
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
            <Chat />
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