import google.generativeai as genai
import os
from pathlib import Path
import mimetypes

class GeminiModel:
    def __init__(self, api_key, model_name="gemini-2.0-flash", repository_path=None):
        """
        Initialize the Gemini model interface.
        
        Args:
            api_key (str): Gemini API key
            model_name (str): Model to use (default: gemini-2.0-flash)
            repository_path (str): Path to the repository files for context
        """
        mimetypes.add_type('text/plain', '.txt')
        self.model_name = model_name
        self.repository_path = repository_path
        
        # Initialize client
        genai.configure(api_key=api_key)
        self.client = genai
        
        # Initialize conversation history
        self.history = []
        self.system_prompt = """You are an expert on everything about the MIT AI Risk Repository project by the MIT futuretech lab. 
        You are an assistant who helps users navigate the repository and provides information on its content and structure.
        Be concise and informative in your responses. If you don't know something, say so instead of making up information.
        """
        
        # Initial system content
        self.history = []
        self.safety_settings = []
        
        # Load repository files if path is provided
        if repository_path:
            self._load_repository_files()
    
    def _load_repository_files(self):
        """Load all repository files into the context"""
        if not self.repository_path or not os.path.exists(self.repository_path):
            print(f"Warning: Repository path {self.repository_path} does not exist")
            return
            
        # No direct file upload in the newer SDK - we'll read and add to prompt
        base_path = Path(self.repository_path)
        self.file_contents = []
        
        for filename in os.listdir(self.repository_path):
            file_path = base_path / filename
            if file_path.name.startswith('.') or not file_path.is_file():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.file_contents.append(f"Content from {filename}:\n{content}")
                print(f"Loaded file: {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
    
    def generate_response(self, user_input, stream=False):
        """
        Generate a response to user input.
        
        Args:
            user_input (str): User's question or message
            stream (bool): Whether to stream the response
            
        Returns:
            str: Model's response
        """
        # Add user input to history
        self.history.append({"role": "user", "parts": [{"text": user_input}]})
        
        # Prepare prompt with system prompt and file contents
        prompt = self.system_prompt
        
        # Add file contents if available (truncated if too long)
        if hasattr(self, 'file_contents') and self.file_contents:
            # Limit to first few files to avoid context limits
            combined_content = "\n\n".join(self.file_contents[:3])
            if len(combined_content) > 8000:
                combined_content = combined_content[:8000] + "...[content truncated]"
            prompt += "\n\nRepository content for reference:\n" + combined_content
        
        # Generate response
        try:
            model = genai.GenerativeModel(model_name=self.model_name)
            
            if stream:
                chat = model.start_chat(history=self.history)
                response = chat.send_message(prompt + "\n\n" + user_input, stream=True)
                
                # For streaming, collect all chunks
                full_response = ""
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        full_response += chunk.text
                
                response_text = full_response
            else:
                chat = model.start_chat(history=self.history)
                response = chat.send_message(prompt + "\n\n" + user_input)
                response_text = response.text
            
            # Add model response to history
            self.history.append({"role": "model", "parts": [{"text": response_text}]})
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def generate_stream(self, prompt, history=None):
        """
        Generate a streaming response to user input.
        
        Args:
            prompt (str): User's question or message
            history (list, optional): Conversation history in the required format
            
        Yields:
            Text chunks from the model's response
        """
        try:
            model = self.client.GenerativeModel(model_name=self.model_name)
            
            # Use provided history or instance history
            chat_history = history if history is not None else self.history
            
            # Initialize chat with history if available
            chat = model.start_chat(history=chat_history)
            
            # Store chunks for saving to history later
            chunks = []
            
            # Send message with streaming enabled
            response = chat.send_message(prompt, stream=True)
            
            # Yield each chunk of text as it arrives
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    # Save chunk for history
                    chunks.append(chunk.text)
                    # Yield chunk for streaming
                    yield chunk.text
            
            # Add message to history - only if using internal history
            if history is None and chunks:
                # Add the prompt as user message
                self.history.append({"role": "user", "parts": [{"text": prompt}]})
                
                # Combine chunks into full response
                full_response = "".join(chunks)
                
                # Add the response as model message
                self.history.append({"role": "model", "parts": [{"text": full_response}]})
                
        except Exception as e:
            error_msg = f"Error generating streaming response: {str(e)}"
            print(error_msg)
            yield error_msg
    
    def reset_conversation(self):
        """Reset the conversation history but keep the system prompt and repository files"""
        # Simply clear the history
        self.history = []
