import { ChatMessagesProps} from '../interfaces/interfaces'

//TODO: UPDATE MESSAGE PROPS
const ChatMessages: React.FC<ChatMessagesProps> = ({ previousMessages }) => {

  return (
    <div className="flex flex-col space-y-2">
      {previousMessages.map((msg) => (
        <div
          key={msg.id}
          className={`p-2 rounded-lg max-w-xl ${
            msg.role === "user" ? "bg-blue-200 self-end" : "bg-gray-200 self-start"
          }`}
        >
          {msg.content}
        </div>
      ))}


    </div>
  );
};

export default ChatMessages;