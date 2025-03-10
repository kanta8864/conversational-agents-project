## Steps needed to run deepseek locally

1. Install Opalla
2. Pull and run whatever model you want to run. Use the ollama pull command to download the DeepSeek model you want to use (e.g. `ollama pull deepseek-r1:8b`) and then run the model with (e.g. `ollama run deepseek-r1:8b`)
3. Start Opalla server with `ollama serve` to expose the model as an API

## Steps to set up Perception Agent

1. Run the front_end.py script.
2. Record audio files by clicking on the microphone button.
3. Give the audio file a name.
4. Upload the audio file by clicking on the green upload button.
5. Install ffmpeg on your laptop

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

6. Run main.py script.
7. See beautiful transcription of your audio file in the console.

Note: Currently we have not fully incorporated the perception module to work with the rest of BingeBuddy. This is because we haven't had the time yet to set up a proper interface for the user to interact with BingeBuddy. This will be done soon!

## Setting Up MongoDB for Development

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your own user-name and password, rest should be kept the same.
3. Run `./start-memory-db.sh` to set up the container and the database.

## TODO: Add more details on how to work with poetry and run modules

Example command to run memory sentinel:

```bash
poetry run python3 -m binge_buddy.memory_sentinel
```

Poetry will run the script inside the virtual environment.

You can also activate the virtualenv created by poetry yourself and then run

```bash
python3 -m binge_buddy.memory_sentinel
```

These essentially do the same thing.

More information on how to set up the project incoming.
