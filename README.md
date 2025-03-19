## Steps to run the temporary frontend

1. Currently our project is still under development, so main.py doesn't work.
2. To run the project please follow steps 1 and 2 mentioned below.
3. After everything has been set up correctly, run the front_end.py script. This can be done by going to `binge_budy` folder and run `python front_end.py`
4. Either type in a text in the text box and press send or record audio files by clicking on the microphone button and upload the file by clicking on the green upload button.
5. You should be able to see our agent's response. Note that our agent is currently very slow and the agent will response in 20-30 seconds or so with its recommendation. PS: It currently only runs on the message log (so short term memory), we haven't had the chance yet to incorporate the long term memory properly.

## 1. Steps needed to run deepseek locally

1. Install Opalla
2. Pull and run whatever model you want to run. Use the ollama pull command to download the DeepSeek model you want to use (e.g. `ollama pull deepseek-r1:8b`) and then run the model with (e.g. `ollama run deepseek-r1:8b`)
3. Start Opalla server with `ollama serve` to expose the model as an API

## 2. Install ffmpeg for perception module

1. Make sure to install ffmpeg on your laptop for perception module to work.

   on Ubuntu or Debian

   sudo apt update && sudo apt install ffmpeg

   on Arch Linux

   sudo pacman -S ffmpeg

   on MacOS using Homebrew (https://brew.sh/)

   brew install ffmpeg

   on Windows using Chocolatey (https://chocolatey.org/)

   choco install ffmpeg

   on Windows using Scoop (https://scoop.sh/)

   scoop install ffmpeg

# The following can be ignored for Lab Assignment 2 because DB is not yet integrated in our pipeline unfortunately.

## 3. Setting Up MongoDB for Development

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your own user-name and password, rest should be kept the same.
3. Run `docker compose up -d` to set up the container and the database.

## TODO: Add more details on how to work with poetry and run modules

Example command to run the semantic workflow:

```bash
poetry run python3 -m binge_buddy.memory_workflow.semantic_workflow
```

Poetry will run the script inside the virtual environment.

You can also activate the virtualenv created by poetry yourself and then run

```bash
python3 -m binge_buddy.memory_workflow.semantic_workflow
```

These essentially do the same thing.

More information on how to set up the project incoming.
