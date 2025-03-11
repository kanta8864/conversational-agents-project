from flask import Flask, request, jsonify, render_template, session
import os
from threading import Thread
from binge_buddy.conversational_agent import ConversationalAgent
from binge_buddy.message import Message
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM
from binge_buddy.perception_agent import PerceptionAgent
from langchain.schema import HumanMessage
import threading
from binge_buddy.memory_handler import app as memory_app


# Set up the Flask app
app = Flask(__name__)
# Initialize global agent and message log (persistent across requests)
llm = OllamaLLM()
agent = PerceptionAgent()
message_log = MessageLog(user_id="user", session_id="session")
conversational_agent = ConversationalAgent(llm, message_log)

sample_memory = {
    "user_id": "kanta_001",
    "name": "Kanta",
    "memories": {
        "LIKES": "Kanta likes sci-fi and action movies but he also enjoys rom-coms.",
        "DISLIKES": "Kanta hates Japanese movies, especially anime.",
        "FAVOURITES": "Kanta's favorite movie is Inception, and he loves Christopher Nolan's films.",
        "GENRE": "Kanta prefers movies with deep storytelling, complex characters, and mind-bending plots.",
    },
    "last_updated": "2024-03-09T12:00:00Z",
}


def run_memory_in_background(response, sample_memory):
    """Runs memory processing in a separate thread after returning response."""
    inputs = {
        "messages": [HumanMessage(content=response)],
        "memories": sample_memory,
    }
    memory_thread = threading.Thread(target=run_memory_module, args=(inputs,))
    memory_thread.daemon = True  # Ensures the thread exits when the main program stops
    memory_thread.start()


def run_memory_module(inputs):
    """Function to process memory pipeline in the background."""
    for output in memory_app.with_config({"run_name": "Memory"}).stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")


@app.route("/")
def index():
    return render_template("front_end.html")


@app.route("/send_message", methods=["POST"])
def handle_user_message():
    """Handle user messages, either text or audio, and process them."""
    data = request.get_json()
    if "text" in data:
        user_message = data["text"]
        print(f"User message received: {user_message}")

        user_message_obj = Message(
            role="user", content=user_message, user_id="user", session_id="session"
        )
        message_log.add_message(user_message_obj)

        # Run agent response generation in the background
        response = conversational_agent.run()
        print(f"Agent response: {response}")

        run_memory_in_background(data["text"], sample_memory)

        return jsonify({"response": response})


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return "No audio file uploaded", 400

    audio_file = request.files["audio"]
    filename = audio_file.filename
    file_path = os.path.join("./audio", filename)

    # Save the audio file to disk
    audio_file.save(file_path)

    # Transcribe the audio file to text
    transcribed_text = agent.transcribe(file_path)

    # Immediately return the transcribed text to the frontend
    return jsonify({"transcribed_text": transcribed_text})


def run_flask():
    """Start the Flask application."""
    app.run(port=5000, host="0.0.0.0", debug=True)


if __name__ == "__main__":
    run_flask()
