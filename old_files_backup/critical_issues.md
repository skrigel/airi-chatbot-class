# Critical Issues and Fixes

## 1. Streaming Response Functionality Broken

**Issue**: The `generate_stream` method in `gemini_model.py` has implementation problems:
   - It tries to iterate over a consumed iterator
   - Its return format doesn't match what the adapter expects

**Fix**: Update the `generate_stream` method in `gemini_model.py`:

```python
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
        
        # Store the text chunks for adding to history later
        chunks = []
        
        # Send message with streaming enabled
        stream_response = chat.send_message(prompt, stream=True)
        
        # Yield each chunk of text as it arrives
        for chunk in stream_response:
            if hasattr(chunk, 'text') and chunk.text:
                # Save the chunk for history
                chunks.append(chunk.text)
                # Yield the chunk text
                yield chunk.text
        
        # Add message to history if we're using instance history
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
```

## 2. Missing Import Errors

**Issue**: The terminal output shows failures to import certain modules like `langchain_google_genai`

**Fix**: Ensure all required packages are properly installed by updating `install_deps.sh` to guarantee these packages are installed:

```bash
# Ensure all critical dependencies are installed
$PIP_CMD install --user flask flask-cors python-dotenv google-generativeai \
    langchain langchain-community langchain-google-genai chromadb tiktoken pydantic rank-bm25 pandas openpyxl
```

## 3. Vector Database Not Being Properly Built

**Issue**: The chatbot is giving generic responses rather than finding specific information in the Excel file

**Fix**: The vector database needs to be properly built during setup. Ensure that:

1. The correct Excel file is being properly processed
2. The rebuild_database.sh script is run
3. The `run.sh` script checks for the database existence

## 4. CORS Support for Frontend

**Issue**: The adapter may be missing proper CORS configuration for the frontend

**Fix**: Ensure CORS is properly enabled in `airi_adapter.py`:

```python
# Create Flask app with CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
```

## 5. Error Handling in Stream Generation

**Issue**: If an error occurs during streaming, it's not being properly handled and may cause the entire response to fail

**Fix**: Improve error handling in streaming code in `airi_adapter.py`:

```python
try:
    # Try to iterate through the response
    for chunk in response_generator:
        try:
            text_chunk = chunk if isinstance(chunk, str) else chunk.text if hasattr(chunk, 'text') else str(chunk)
            yield json.dumps(text_chunk) + '\n'
            complete_response += text_chunk
            time.sleep(0.05)
        except Exception as chunk_err:
            logger.error(f"Error processing chunk: {str(chunk_err)}")
            # Continue processing despite error with a chunk
            continue
except Exception as stream_err:
    logger.error(f"Error streaming response: {str(stream_err)}")
    # Fall back to non-streaming response
    response = gemini_model.generate_response(prompt, stream=False)
    yield json.dumps(response) + '\n'
    complete_response = response
```

## 6. Improve Vector Store Efficiency

**Issue**: The vector store might not be properly searching the Excel file content

**Fix**: Add explicit handling for the specific Excel file by updating the rebuild_database.sh script:

```bash
echo "Adding special handling for the main AI Risk Repository Excel file..."
if [ -f "info_files/The_AI_Risk_Repository_V3_26_03_2025.xlsx" ]; then
  echo "Found main Excel repository file. Ensuring it's prioritized..."
  # Create a special flag file to tell the vector store to prioritize this file
  echo "excel_priority=The_AI_Risk_Repository_V3_26_03_2025.xlsx" > info_files/.repository_config
fi
```

## Implementation Plan

1. Fix the `generate_stream` method in `gemini_model.py`
2. Update the dependencies installation in `install_deps.sh`
3. Improve the vector database building in `setup.sh` and `rebuild_database.sh` 
4. Add error handling to the streaming response in `airi_adapter.py`
5. Ensure CORS is correctly configured
6. Test the streaming functionality and database queries