from transformers import pipeline
from binge_buddy.conversational_agent_manager import ConversationalAgentManager
from binge_buddy.message.human_message import HumanMessage

class SentimentAnalyzer:
    def __init__(self, llm):
        self.emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        self.conversational_agent_manager = ConversationalAgentManager(llm)

    def extract_emotion(self, message:str, user_id, session_id) -> str:
        result = self.emotion_analyzer(message)
        agent = self.conversational_agent_manager.get_agent(user_id=user_id, session_id=session_id)
        print(result)
        # todo: do something with the emotion result
        message = HumanMessage(content=message, user_id=user_id, session_id=session_id)
        return agent.process_message(message)