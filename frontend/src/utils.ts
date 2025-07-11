// import { EventSourceParserStream } from 'eventsource-parser/stream';
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

//TODO: implement streaming

// export async function* parseSSEStream(stream) {
//   const sseStream = stream
//     .pipeThrough(new TextDecoderStream())
//     .pipeThrough(new EventSourceParserStream())
  
//   for await (const chunk of sseStream) {
//     if (chunk.type === 'event') {
//       yield chunk.data;
//     } 
//   }
// }