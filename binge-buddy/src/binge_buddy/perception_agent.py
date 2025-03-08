import whisper

from transformers import pipeline

class PerceptionAgent:
    def __init__(self):
        # Load the pre-trained model for English
        self.model = whisper.load_model("small.en")

    def transcribe(self, audio_file):
        # Load the audio file
        audio = whisper.load_audio(audio_file)
        # trim the audio to 30 seconds (just for testing, comment this out for production)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # decode the audio
        options = whisper.DecodingOptions(language="en")
        result = whisper.decode(self.model, mel, options)

        return result.text

    def extract_emotion(self, message):
        emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        result = emotion_analyzer(message)
        return result