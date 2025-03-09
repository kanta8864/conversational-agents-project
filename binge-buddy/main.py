from pathlib import Path
from binge_buddy.perception_agent import PerceptionAgent
from binge_buddy.memory_handler import app

def run_perception():
    # Make instance of PerceptionAgent
    agent = PerceptionAgent()

    # Get the path to the audio folder using pathlib
    audio_folder = Path("audio")
    result = None

    # Transcribe audio for each file in the audio folder
    for (
        audio_file
    ) in audio_folder.iterdir():  # iterdir() gives us each file in the folder
        if audio_file.is_file():  # Check if it's a file (not a directory)
            result = agent.transcribe(audio_file)
            print(result)

            # Delete the audio file after transcribing
            # audio_file.unlink()  # unlink() deletes the file

    return result


def main():
    # Example message to analyze
    current_message = "I love watching sci-fi movies like The Matrix!"

    # Attempt to pass perception message
    # current_message = run_perception()

    # Run the whole memory pipeline with langgraph
    for output in app.with_config({"run_name": "Memory"}).stream(current_message):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

if __name__ == "__main__":
    main()
