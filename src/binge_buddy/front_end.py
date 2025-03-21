import time
import os
from binge_buddy.conversational_agent_manager import ConversationalAgentManager
from binge_buddy.memory_db import MemoryDB
from binge_buddy.memory_handler import EpisodicMemoryHandler, SemanticMemoryHandler
from binge_buddy.perception.audito_transcriber import AudioTranscriber
from binge_buddy.perception.sentiment_analyzer import SentimentAnalyzer
from flask import Flask, request, jsonify, render_template
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM


# Set up the Flask app
app = Flask(__name__)
llm = OllamaLLM()
sentiment_analyzer = SentimentAnalyzer()
memory_db = MemoryDB()
mode = "semantic"
if mode == "semantic":
    memory_handler = SemanticMemoryHandler(memory_db)
else:
    memory_handler = EpisodicMemoryHandler(memory_db)
message_log = MessageLog(
    user_id="user", session_id="session", memory_handler=memory_handler, mode=mode
)
conversational_agent_manager = ConversationalAgentManager(
    llm, message_log, memory_handler
)
audio_transcriber = AudioTranscriber()


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


@app.route("/")
def index():
    return render_template("front_end.html")


@app.route("/send_message", methods=["POST"])
def handle_user_message():
    data = request.get_json()
    if "text" in data:
        user_message = data["text"]
        print(f"User message received: {user_message}")
        # todo: update userID and sessionID
        message = sentiment_analyzer.extract_emotion(
            user_message, "userId", "sessionId"
        )
        ca = conversational_agent_manager.get_agent("userId", "sessionId")
        response = ca.process_message(message)
        return jsonify({"response": response})


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]

    os.makedirs("./audio", exist_ok=True)

    filename = f"{int(time.time())}_{audio_file.filename}"
    file_path = os.path.join("./audio", filename)

    # Save the audio file to disk
    audio_file.save(file_path)

    transcribed_text = audio_transcriber.transcribe(file_path)

    os.remove(file_path)

    return jsonify({"transcribed_text": transcribed_text})


def run_flask():
    """Start the Flask application."""
    app.run(port=5000, host="0.0.0.0", debug=True)


if __name__ == "__main__":
    run_flask()
