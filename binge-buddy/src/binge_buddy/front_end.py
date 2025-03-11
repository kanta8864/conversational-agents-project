import flask
from flask import Flask, request, jsonify, render_template
import os

from binge_buddy.conversational_agent import ConversationalAgent
from binge_buddy.message import Message
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM

# Set up the Flask app
app = Flask(__name__)

# Initialize global agent and message log (persistent across requests)
llm = OllamaLLM()
message_log = MessageLog(user_id="user", session_id="session")
conversational_agent = ConversationalAgent(llm, message_log)


@app.route("/")
def index():
    return render_template("front_end.html")


@app.route("/send_message", methods=["POST"])
def handle_user_message():
    """Handles incoming user messages and returns agent responses."""
    data = request.get_json()

    if "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]
    print(f"User message received: {user_message}")

    # Create a Message object for the user input
    user_message_obj = Message(
        role="user", content=user_message, user_id="user", session_id="session"
    )
    message_log.add_message(user_message_obj)

    # Use the globally initialized agent to generate a response
    response = conversational_agent.run()
    print(f"Agent response: {response}")

    return jsonify({"response": response})


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return "No audio file uploaded", 400

    audio_file = request.files["audio"]
    filename = audio_file.filename
    # Save the audio file with the custom name provided by the user
    audio_file.save(os.path.join("../../audio", filename))

    return "Audio uploaded successfully", 200


def run_flask():
    app.run(port=5000, host="0.0.0.0", debug=True)


if __name__ == "__main__":
    run_flask()
