from google import genai
from google.genai import types
import os
# import load_env()
from pathlib import Path

import mimetypes

mimetypes.add_type('text/plain', '.txt')
GEMINI_API_KEY='AIzaSyDosr02ZzhOptpHts-zFcyZcUxwpVSjszI'

class Gemini:

    @staticmethod
    def generate(user_input, history=None):
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(api_version='v1alpha'),
            
        )

        model = "gemini-2.0-flash"


        if history is None or not isinstance(history, list):
            history = []
           
        # Always start with system prompt only once
        if not history:
            history.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="""You are an expert on everything about the MIT AI Risk Repository project by the MIT FutureTech Lab. Pretend you are an assistant helping customers navigate the repository and providing information about its content and structure. Be concise.""")
                    ]
                )

            )
            myfile = client.files.upload(file="info_files/airi.txt")
            history.append(myfile)
            

           
        # 🛡 FIX: Only accept user_input if it's a valid string
        if not isinstance(user_input, str):
            raise ValueError("User input must be a string.")


        # Add the new user message
        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        )

        response = client.models.generate_content(
                    model=model,
                    contents=history
                )

        bot_reply = response.candidates[0].content.parts[0].text
                # Add bot reply into history
        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=bot_reply)]
            )
        )

        return bot_reply, history
