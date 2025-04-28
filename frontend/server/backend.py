# from flask import Flask, request, jsonify
##
# convo_details = request.get_json()

##
from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context
from flask_cors import CORS
from chat import Gemini


# Front-end SHOULD NOT LOOK AT HOW BACKEND WORKS



# - Only need to card about how data is formatted!! 
# i.e. 

# {message: …,

# Sender:…, (only for user accounts)
#role: ...,

# Message: history, (metadata!!) 

# }


# message = convo_details.get("provider", "gpt")
chat_history = []  # at top level of server file

app = Flask(__name__)
CORS(app)

@app.route("/api/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/api/v1/sendMessage", methods=['POST'])
def chat():
    global chat_history
    message = request.json.get('message')
    if message:

        ##Place holder

        # response = chatbot.get_response(message)
        print("MESSAGE",message)
        # response = Gemini.generate()
        bot_reply, chat_history = Gemini.generate(message, chat_history)
        return jsonify({'response': bot_reply})
    else:
        return jsonify({'error': 'No message provided'}), 400
    

# prompt history
## CHOOSE FORMAT - more modular, more robust
# {
# prompt: "",
# history: [
# {sender: "", prompt: ""},
# ]
# }

# some sense of user v. AI chat

#i.e.

# {
# prompt: "",
# history: [
# {sender: "", prompt: ""},
# ]
# }

# reuse boolean to format colors - i.e. isBlue --> AI
# quick heuristics--> more compartmentalize --> easier to optimize

if __name__ == '__main__':
    app.run(debug=True)