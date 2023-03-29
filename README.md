# GPTranscribe

Welcome to GPTranscribe(for now)! This python application(for now) allows for transcription of system audio(ie a meeting or youtube video), extraction of important terms and GPT driven definitions for the terms you used to google during a meeting or video. Think of this as the equivelant of left clicking text and searching for a definition but for system audio! 

## Setup

GPTranscribe requires 3 main dependencies
1. python 3.7-3.9
2. An openAI API key
3. blackhole 2 Channel audio driver

Assuming the user has 1, 2 and a cloned repo follow these steps for setup:

1. Install audio driver dependencies
   - [blackhole2ch](https://github.com/ExistentialAudio/BlackHole) `brew install blackhole2-ch`
   - PortAudio `brew install portaudio`
2. Create a multi-audio device for blackhole + your current/desired output. Check [here](https://github.com/ExistentialAudio/BlackHole/wiki/Multi-Output-Device) for instructions.
3. Create a `.env` file with `OPENAI_KEY=your_api_key`
4. Install python requirements `pip install -r requirements.txt`
5. Ensure you're using the multi-audio device from (2) and run `python smart_transcribe.py`
6. *optional: For a basic user interface run `python server.py` alongside 