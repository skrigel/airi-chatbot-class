export interface message {
    content:string;
    role:string;
    id:string;
}
export interface ChatInputProps {
    question: string;
    setQuestion: (question: string) => void;
    onSubmit: (text?: string) => void;
    isLoading: boolean;
}

export interface ChatMessagesProps {
    previousMessages: message[];
  }
  