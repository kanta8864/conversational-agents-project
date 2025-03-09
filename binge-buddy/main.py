from pathlib import Path
from binge_buddy.perception_agent import PerceptionAgent
from binge_buddy.memory_handler import app
from langchain_core.messages import HumanMessage

def run_perception():
    # Make instance of PerceptionAgent
    agent = PerceptionAgent()

    # Get the path to the audio folder using pathlib
    audio_folder = Path("src/binge_buddy/audio")
    result = []
    emotion_result = []

    # Transcribe audio for each file in the audio folder
    for (
        audio_file
    ) in audio_folder.iterdir():  # iterdir() gives us each file in the folder
        if audio_file.is_file():  # Check if it's a file (not a directory)
            result_item = agent.transcribe(audio_file)
            result.append(result_item)
            # print(result)

            # Extract emotion from the transcribed message
            emotion_result_item = agent.extract_emotion(result_item)
            emotion_result.append(emotion_result_item)
            # print(emotion_result)

            # Delete the audio file after transcribing
            # audio_file.unlink()  # unlink() deletes the file

    return result, emotion_result


def main():
    # Example message to analyze
    current_message = "I love watching sci-fi movies like The Matrix!"

    # Attempt to pass perception message
    # current_message = run_perception()

    # Ensure the input is structured correctly, with 'messages' being a list
    inputs = {
        "messages": [HumanMessage(content=current_message)],
    }

    # Run the whole memory pipeline with langgraph
    for output in app.with_config({"run_name": "Memory"}).stream(inputs):
        # Output from the graph nodes (app stream processing)
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

if __name__ == "__main__":
    # main()
    result, emotion_result = run_perception()
    print(result)
    print(emotion_result)