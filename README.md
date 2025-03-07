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
	# on Ubuntu or Debian
	sudo apt update && sudo apt install ffmpeg

	# on Arch Linux
	sudo pacman -S ffmpeg

	# on MacOS using Homebrew (https://brew.sh/)
	brew install ffmpeg

	# on Windows using Chocolatey (https://chocolatey.org/)
	choco install ffmpeg

	# on Windows using Scoop (https://scoop.sh/)
	scoop install ffmpeg
5. Run main.py script.
6. See beautiful transcription of your audio file in the console.