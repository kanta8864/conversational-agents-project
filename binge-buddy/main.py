# from pathlib import Path
# import sys
# from binge_buddy.memory_db import MemoryDB
# from binge_buddy.memory_handler import app
# from binge_buddy.message import Message
# from binge_buddy.message_log import MessageLog
# from binge_buddy.perception_agent import PerceptionAgent
# from binge_buddy.semantic_agent import SemanticAgent
# from binge_buddy.ollama import OllamaLLM


# def run_perception():
#     # Make instance of PerceptionAgent
#     agent = PerceptionAgent()

#     # Get the path to the audio folder using pathlib
#     audio_folder = Path("src/binge_buddy/audio")
#     result = []
#     emotion_result = []

#     # Transcribe audio for each file in the audio folder
#     for (
#         audio_file
#     ) in audio_folder.iterdir():  # iterdir() gives us each file in the folder
#         if audio_file.is_file():  # Check if it's a file (not a directory)
#             result_item = agent.transcribe(audio_file)
#             result.append(result_item)
#             # print(result)

#             # Extract emotion from the transcribed message
#             emotion_result_item = agent.extract_emotion(result_item)
#             emotion_result.append(emotion_result_item)
#             # print(emotion_result)

#             # Delete the audio file after transcribing
#             # audio_file.unlink()  # unlink() deletes the file

#     return result, emotion_result


# def main():
#     # Attempt to connect to db
#     db = MemoryDB()
#     sample_memory = {
#         "user_id": "kanta_001",
#         "name": "Kanta",
#         "memories": {
#             "LIKES": "Kanta likes sci-fi and action movies but he also enjoys rom-coms.",
#             "DISLIKES": "Kanta hates Japanese movies, especially anime.",
#             "FAVOURITES": "Kanta's favorite movie is Inception, and he loves Christopher Nolan's films.",
#             "GENRE": "Kanta prefers movies with deep storytelling, complex characters, and mind-bending plots.",
#         },
#         "last_updated": "2024-03-09T12:00:00Z",
#     }

#     db.insert_one(collection_name="memories", data=sample_memory)

#     memories = db.find_one(collection_name="memories", query={"name": "Kanta"})

#     llm = OllamaLLM()
#     message_log = MessageLog(user_id="user", session_id="session")

#     semantic_agent = SemanticAgent(llm, message_log)

#     while True:

#         user_message = input("Type your input:")

#         if user_message == "/bye":
#             sys.exit()

#         user_message = Message(
#             role="user", content=user_message, user_id="user", session_id="session"
#         )

#         message_log.add_message(user_message)

#         response = semantic_agent.run()

#         print("\n Response:", response, "\n")

#     # result, emotion_result = run_perception()

#     # for (message,emotion) in zip(result, emotion_result):
#     #     # Ensure the input is structured correctly, with 'messages' being a list
#     #     inputs = {
#     #         "messages": [HumanMessage(content=message)],
#     #         "memories": memories["memories"],
#     #     }
#     #     # Run the whole memory pipeline with langgraph
#     #     for output in app.with_config({"run_name": "Memory"}).stream(inputs):
#     #         # Output from the graph nodes (app stream processing)
#     #         for key, value in output.items():
#     #             print(f"Output from node '{key}':")
#     #             print("---")
#     #             print(value)
#     #         print("\n---\n")


# if __name__ == "__main__":
#     main()
