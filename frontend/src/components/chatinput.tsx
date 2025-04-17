import { Textarea } from "../ui/textarea";
import { cx } from 'classix';
// import { Button } from "../ui/button";
import { ArrowUpIcon } from "./icons"
import { toast } from 'sonner';
// import { useEffect, useState } from 'react';
import { ChatInputProps } from '../interfaces/interfaces'

//functions to reformat input --> backend??
// flask integrations with react

export const ChatInput = ({ question, setQuestion, onSubmit, isLoading }: ChatInputProps) => {

 
    return(
    <div className="relative w-full flex flex-col gap-4">
     

        <Textarea
        placeholder="Send a message..."
        className={cx(
            'min-h-[24px] max-h-[calc(75dvh)] overflow-hidden resize-none rounded-xl text-base bg-muted',
        )}
        value={question}
        //updates the question inside the box
        onChange={(e) => setQuestion(e.target.value)}
        //press enter --> goes to backend
        onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();

                if (isLoading) {
                    toast.error('Please wait for the model to finish its response!');
                } else {
                    // setShowSuggestions(false);
                    onSubmit();
                }
            }
        }}
        rows={3}
        autoFocus
        />

        <button 
            className="rounded-full p-1.5 h-fit absolute bottom-2 right-2 m-0.5 border dark:border-zinc-600"
            onClick={() => onSubmit(question)}
            disabled={question.length === 0}
        >
            <ArrowUpIcon size={14} />
        </button>
    </div>
    );
}