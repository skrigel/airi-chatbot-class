import google.generativeai as genai
from google.genai import types
import os
import dotenv
from pathlib import Path

import mimetypes

mimetypes.add_type('text/plain', '.txt')
dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

class Gemini:

    generation_config= {
        "temperature": .15,
        "top_p": .95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain"
    }

    # files = [{'file_uri':'info_files/airi.txt', 'mime_type': 'plain/text'}]

    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)

        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",  # use available model like "gemini-1.5-pro" or "gemini-1.5-flash"
            generation_config = self.generation_config,
            system_instruction="""You are an expert on everything about the MIT AI Risk Repository project by the MIT FutureTech Lab. Pretend you are an assistant helping customers navigate the repository and providing information about its content and structure. Be concise."""
        )

        self.file = genai.upload_file(
            path="info_files/airi.txt",
            mime_type="text/plain"
        )


    def generate(self, user_input, history=None):

        if not history:
            history = []
            history.append({
                    "role": "user",
                    "parts": [
                        *[{
                            "file_data": {
                                "mime_type": self.file.mime_type,
                                "file_uri": self.file.uri
                            }
                        }] ,
                        "Please read and use the attached file to inform your responses."
                    ]
                }) 
            
        if not isinstance(user_input, str):
            raise ValueError("Input must be a string.")
        

        history.append(
                {"role": "user", "parts":user_input})

        chat_session = self.model.start_chat(
            history=history
        )

        response = chat_session.send_message(user_input)

        history.append({"role": "model", "parts":response.text})


        return response.text, history
       