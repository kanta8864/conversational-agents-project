from perception_agent import PerceptionAgent

if __name__ == "__main__":
    agent = PerceptionAgent()
    result = agent.transcribe("ukfood.mp3")
    print(result)