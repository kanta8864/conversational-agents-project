import os
from perception_agent import PerceptionAgent
from langchain_core.messages import HumanMessage
from memory_handler import app

if __name__ == "__main__":
    # Make instance of PerceptionAgent
    agent = PerceptionAgent()
    # Transcribe audio for each file in the audio folder
    for audio_file in os.listdir("audio"):
        result = agent.transcribe(f"audio/{audio_file}")
        print(result)

        # Delete the audio file after transcribing
        os.remove(f"audio/{audio_file}")


        # Run the graph 
        inputs = {
            "messages": [HumanMessage(content=result)],
        }

        for output in app.with_config({"run_name": "Memory"}).stream(inputs):
            # stream() yields dictionaries with output keyed by node name
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
            print("\n---\n")
    
