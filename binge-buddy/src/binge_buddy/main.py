import os
from perception_agent import PerceptionAgent

if __name__ == "__main__":
    # Make instance of PerceptionAgent
    agent = PerceptionAgent()
    # Transcribe audio for each file in the audio folder
    for audio_file in os.listdir("audio"):
        result = agent.transcribe(f"audio/{audio_file}")
        print(result)
        # Delete the audio file after transcribing
        # os.remove(f"audio/{audio_file}")