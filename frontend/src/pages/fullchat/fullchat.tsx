import { Header } from "../../components/header";
import { Chat } from '../chat/chat';



export function FullChat() {
  return (
    <div className="relative h-dvh bg-gray-50 text-black flex flex-col">
      <Header />
      <div className="flex-1 p-10 overflow-hidden">
        <Chat />
      </div>
    </div>
  );
}
