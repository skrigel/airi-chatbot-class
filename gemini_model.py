from google import genai
from google.genai import types
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
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version='v1alpha')
        )
        
        # Initialize conversation history
        self.history = []
        self.system_prompt = """You are an expert on everything about the MIT AI Risk Repository project by the MIT futuretech lab. 
        You are an assistant who helps users navigate the repository and provides information on its content and structure.
        Be concise and informative in your responses. If you don't know something, say so instead of making up information.
        """
        
        # Initial system content
        self.history = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self.system_prompt)]
            )
        ]
        
        # Load repository files if path is provided
        if repository_path:
            self._load_repository_files()
    
    def _load_repository_files(self):
        """Load all repository files into the context"""
        if not self.repository_path or not os.path.exists(self.repository_path):
            print(f"Warning: Repository path {self.repository_path} does not exist")
            return
            
        base_path = Path(self.repository_path)
        for filename in os.listdir(self.repository_path):
            file_path = base_path / filename
            if file_path.name.startswith('.') or not file_path.is_file():
                continue
                
            try:
                uploaded_file = self.client.files.upload(file=file_path)
                self.history.append(uploaded_file)
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
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        )
        
        # Generate response
        try:
            if stream:
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=self.history
                )
                
                # For streaming, we'll collect all chunks
                full_response = ""
                for chunk in response_stream:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                chunk_text = candidate.content.parts[0].text
                                full_response += chunk_text
                                # When streaming, you'd typically yield each chunk
                                # Here we're just collecting them
                
                response_text = full_response
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self.history
                )
                response_text = response.candidates[0].content.parts[0].text
            
            # Add model response to history
            self.history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=response_text)]
                )
            )
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def reset_conversation(self):
        """Reset the conversation history but keep the system prompt and repository files"""
        # Save uploaded files
        uploaded_files = [content for content in self.history if not hasattr(content, 'role')]
        
        # Reset history with system prompt
        self.history = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self.system_prompt)]
            )
        ]
        
        # Re-add uploaded files
        self.history.extend(uploaded_files)