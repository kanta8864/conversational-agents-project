from pathlib import Path

from binge_buddy.memory_sentinel import MemorySentinel
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM
from binge_buddy.perception_agent import PerceptionAgent


def run_perception():
    # Make instance of PerceptionAgent
    agent = PerceptionAgent()

    # Get the path to the audio folder using pathlib
    audio_folder = Path("src/binge_buddy/audio")
    result = None

    # Transcribe audio for each file in the audio folder
    for (
        audio_file
    ) in audio_folder.iterdir():  # iterdir() gives us each file in the folder
        if audio_file.is_file():  # Check if it's a file (not a directory)
            result = agent.transcribe(audio_file)
            # print(result)

            # Delete the audio file after transcribing
            # audio_file.unlink()  # unlink() deletes the file

    # Extract emotion from the transcribed message
    emotion_result = agent.extract_emotion(result)
    # print(emotion_result)

    return result, emotion_result


def main():
    # Example setup for the memory sentinel
    llm = OllamaLLM()  # Or any other LLM model
    message_log = MessageLog(session_id="12345", user_id="user123")

    memory_sentinel = MemorySentinel(
        long_term_memory_db=None, message_log=message_log, llm=llm
    )

    # Example message to analyze
    current_message = "I love watching sci-fi movies like The Matrix!"
    result = memory_sentinel.analyze_message(current_message)
    print(f"Store this information? {result}")

    # Attempt to pass perception message
    current_message, current_emotion = run_perception()
    result = memory_sentinel.analyze_message(
        current_message if current_message else "Hey!"
    )
    print(f"Store this information? {result}")


if __name__ == "__main__":
    main()