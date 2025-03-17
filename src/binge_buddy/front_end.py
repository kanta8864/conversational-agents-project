from binge_buddy.memory_db import MemoryDB
from binge_buddy.perception.sentiment_analyzer import SentimentAnalyzer
from flask import Flask, request, jsonify, render_template
from threading import Thread
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM



# Set up the Flask app
app = Flask(__name__)
# Initialize global agent and message log (persistent across requests)
llm = OllamaLLM()
sentiment_analyzer = SentimentAnalyzer(llm)
message_log = MessageLog(user_id="user", session_id="session")
memory_db = MemoryDB()

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
        response = sentiment_analyzer.extract_emotion(user_message, "userId", "sessionId")
        return jsonify({"response": response})


# @app.route("/upload", methods=["POST"])
# def upload_audio():
#     if "audio" not in request.files:
#         return "No audio file uploaded", 400

#     audio_file = request.files["audio"]
#     filename = audio_file.filename
#     file_path = os.path.join("./audio", filename)

#     # Save the audio file to disk
#     audio_file.save(file_path)

#     # Transcribe the audio file to text
#     transcribed_text = agent.transcribe(file_path)

#     # Immediately return the transcribed text to the frontend
#     return jsonify({"transcribed_text": transcribed_text})


def run_flask():
    """Start the Flask application."""
    app.run(port=5000, host="0.0.0.0", debug=True)


if __name__ == "__main__":
    run_flask()
