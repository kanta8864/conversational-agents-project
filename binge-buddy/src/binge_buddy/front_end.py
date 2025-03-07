import flask

# Set up the Flask app
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('front_end.html')


@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in flask.request.files:
        return "No audio file uploaded", 400
    else:
        audio_file = flask.request.files['audio']
        audio_file.save("audio/recorded_audio.wav")
        return "Audio uploaded successfully", 200



def run_flask():
    app.run(port=5000, host='0.0.0.0', debug=True)

if __name__ == "__main__":
    run_flask()