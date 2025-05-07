from google import genai
from google.genai import types
import os
from pathlib import Path

import mimetypes

mimetypes.add_type('text/plain', '.txt')
GEMINI_API_KEY='AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI'

def generate():
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version='v1alpha')
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""You are an expert on everything about the MIT AI Risk Respository project by the MIT futuretech lab. Pretend you are an assistant who is helping customers navigate the repository and who can provide information on its content and structure. Be as concise as you can. 
                                     If there are any files attached, use the info within to further inform your expertise."""),
            ],
        ),
    ]
    directory = '/Users/arjunchidrawar/Desktop/RAG_stuff/airi-chatbot-class/info_files'
    base_path = Path(directory)
    for filename in os.listdir(directory):
        file_path = base_path / filename
        if file_path.name.startswith('.') or not file_path.is_file():
            continue

        uploaded_file = client.files.upload(file=file_path)
        contents.append(uploaded_file)
        

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        )

        response = client.models.generate_content(
            model=model,
            contents=contents
        )
        bot_reply = response.candidates[0].content.parts[0].text
        print("Bot:", bot_reply)

        contents.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=bot_reply)]
            )
        )

if __name__ == "__main__":
    generate()

