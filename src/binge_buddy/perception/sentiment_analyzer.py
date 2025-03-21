from transformers import pipeline
from binge_buddy.message import UserMessage


class SentimentAnalyzer:
    def __init__(self):
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
        )

    def extract_emotion(self, message: str, user_id, session_id) -> str:
        result = self.emotion_analyzer(message)
        user_message = UserMessage(
            content=message, user_id=user_id, session_id=session_id
        )
        return user_message
