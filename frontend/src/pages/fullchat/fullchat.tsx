import { useState, useRef } from 'react';
import { Chat } from '../chat/chat';
import { Header } from '../../components/header';
import { message } from '../../interfaces/interfaces';
import { v4 as uuidv4 } from 'uuid';

const API_URL = '';

const WELCOME: message = {
  content: "Hi! I'm your AI assistant to help you navigate the AI Risk repository. How can I help you today?",
  role: 'assistant',
  id: uuidv4(),
};

export function FullChat() {
  const [previousMessages, setPreviousMessages] = useState<message[]>([]);
  const [currentMessage, setCurrentMessage] = useState<message | null>(WELCOME);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messageHandlerRef = useRef<((event: MessageEvent) => void) | null>(null);

  const [domain, setDomain] = useState('');
  const [suggestedUseCases, setSuggestedUseCases] = useState<string[]>([]);
  const [relatedDocuments, setRelatedDocuments] = useState<{ title: string; url: string }[]>([]);

  const cleanupMessageHandler = () => {
    if (messageHandlerRef.current) {
      messageHandlerRef.current = null;
    }
  };

  async function handleSubmit(text?: string) {
    if (isLoading) return;
    const messageText = text;
    if (!messageText || !messageText.trim()) return;

    const userMessage: message = { content: messageText, role: 'user', id: uuidv4() };
    setPreviousMessages((prev) => [...prev, userMessage]);

    const loadingMessage: message = { content: 'Loading...', role: 'assistant', id: 'loading' };
    setCurrentMessage(loadingMessage);

    try {
      const stream = await fetch(`${API_URL}api/v1/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText }),
      });

      if (!stream.body) throw new Error('Failed!!');

      const reader = stream.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let buffer = '';
      let accumulatedText = '';

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        buffer += decoder.decode(value, { stream: !done });

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const parsed = JSON.parse(line);
              if (parsed.related_documents) {
                setRelatedDocuments(parsed.related_documents);
              } else {
                accumulatedText += parsed;
                const currMess: message = { content: accumulatedText, role: 'assistant', id: uuidv4() };
                setCurrentMessage(currMess);
              }
            } catch (err) {
              console.error('Error parsing line:', line, err);
            }
          }
        }
      }

      const botMessage: message = {
        content: accumulatedText,
        role: 'assistant',
        id: uuidv4(),
      };

      setPreviousMessages((prev) => [...prev, botMessage]);
      setCurrentMessage(null);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: message = {
        content: 'Error talking to server.',
        role: 'assistant',
        id: uuidv4(),
      };
      setPreviousMessages((prev) => [...prev, errorMessage]);
      setCurrentMessage(null);
    } finally {
      setIsLoading(false);
      cleanupMessageHandler();
    }
  }

  const handleDomainSubmit = async () => {
    if (!domain.trim()) return;

    try {
      const response = await fetch(`${API_URL}api/v1/use_cases`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch use cases');
      }

      const data = await response.json();
      setSuggestedUseCases(data.use_cases || []);
    } catch (error) {
      console.error('Error fetching use cases:', error);
      setSuggestedUseCases(['Error fetching use cases.']);
    }
  };

  const defaultUseCases = ['Medical chatbot', 'Customer service agent', 'Model risk review'];

  return (
    <div className="min-h-screen bg-gray-50 text-black flex flex-col">
      <Header />

      <main className="flex-1 p-6 xl:p-10 grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Section – Text + Chat + Example Inputs */}
        <div className="xl:col-span-2 space-y-6">
          {/* 0. Text Explainer */}
          <section className="space-y-2">
            <h2 className="text-2xl font-bold text-gray-800">AI Risk Assistant</h2>
            <p className="text-sm text-gray-600 leading-relaxed max-w-2xl">
              This assistant helps you explore and assess AI risk factors, drawing from resources like the IBM Risk Atlas. You can
              ask about governance frameworks, benchmarks, mitigations, and more.
            </p>
          </section>

          {/* Chat Component */}
          <div className="bg-white rounded-xl shadow border p-4 h-[80vh] overflow-y-auto">
            <Chat
              previousMessages={previousMessages}
              currentMessage={currentMessage}
              handleSubmit={handleSubmit}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* 1. Right Panel – Related Docs, Frameworks, Benchmarks */}
        <aside className="space-y-6 hidden xl:block">
          <div className="bg-white border rounded-xl p-4 shadow-sm">
            <h3 className="text-md font-semibold mb-2">Related Documents</h3>
            <ul className="text-sm text-gray-600 space-y-2">
              {relatedDocuments.length > 0 ? (
                relatedDocuments.map((doc, index) => (
                  <li key={index}>
                    <a href={doc.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">
                      {doc.title}
                    </a>
                  </li>
                ))
              ) : (
                <li>No documents yet. Submit a request to populate this area.</li>
              )}
            </ul>
          </div>

          <div className="bg-white border rounded-xl p-4 shadow-sm">
            {/** TODO: attach a model to suggest use for use cases for different types of users */}
            <h3 className="text-md font-semibold mb-2">Example Use Cases</h3>
            <div className="flex flex-col sm:flex-row gap-2 items-start sm:items-end">
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-600 mb-1">Enter your domain or field:</label>
                <input
                  type="text"
                  placeholder="e.g. education, finance, healthcare"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-red-400"
                />
              </div>
              <button
                onClick={handleDomainSubmit}
                className="bg-red-500 text-white px-4 py-2 rounded-md hover:bg-red-600 transition text-sm"
              >
                Generate Use Cases
              </button>
            </div>

            {/* Use Cases */}
            <div className="space-y-2">
              <h4 className="text-sm text-gray-600 font-semibold">Suggested Use Cases</h4>
              <div className="flex flex-wrap gap-2">
                {defaultUseCases.map((uc) => (
                  <button
                    key={uc}
                    onClick={() => handleSubmit(uc)}
                    className="bg-red-100 text-red-700 text-sm px-3 py-1 rounded-full hover:bg-red-200 transition"
                  >
                    {uc}
                  </button>
                ))}
                {suggestedUseCases.map((uc, i) => (
                  <button
                    key={`suggested-${i}`}
                    onClick={() => handleSubmit(uc)}
                    className="bg-gray-100 text-gray-700 text-sm px-3 py-1 rounded-full hover:bg-gray-200 transition"
                  >
                    {uc}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="bg-white border rounded-xl p-4 shadow-sm">
            <h3 className="text-md font-semibold mb-2">Request Types</h3>
            <span className="text-gray-500">Try asking about:</span>
            {['Risk Taxonomies', 'Benchmarks', 'Mitigations'].map((type) => (
              <button
                key={type}
                onClick={() => handleSubmit(type)}
                className="bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-full"
              >
                {type}
              </button>
            ))}
          </div>
        </aside>
      </main>
    </div>
  );
}