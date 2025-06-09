# RadarX: Emotion-Aware Voice Assistant with Raspberry Pi Integration

## Project Overview

**RadarX** is a PC-based voice assistant pipeline that:
1. Records user speech.
2. Converts it to text using Speech-to-Text (STT).
3. Performs emotion classification and response generation using an LLM-based API.
4. Synthesizes the generated response back into speech (TTS).
5. Sends the synthesized audio to a Raspberry Pi device.

This system is ideal for building interactive, emotion-aware voice agents that respond naturally and empathetically.

---

## Core Components

- `record.py`: Handles microphone input and saves as `input.wav`.
- `stt.py`: Converts recorded speech into text using a speech recognition model.
- `emotion_client.py`: Sends the transcribed text to an emotion-aware LLM server and receives emotion + response.
- `tts.py`: Converts the generated text response into an audio file (`response.wav`) using TTS.
- `transfer.py`: Transfers the audio file to a Raspberry Pi via socket or network protocol.
- `config.py`: Stores configuration parameters like IP address, ports, or API paths.
- `main.py`: Pipeline integration script to run all steps in sequence.

---

## Required Python Libraries

To run this project, make sure the following Python packages are installed:


- pip install sounddevice
- pip install scipy
- pip install openai-whisper
- pip install torch
- pip install requests
- pip install pydub

- Additionally, FFmpeg must be installed and added to your system PATH to support audio format conversion with pydub.

## Run the Assistant
python main.py

- Output
Extracted text and emotion are printed in the terminal.
Audio response is saved as response.wav.
The response audio is automatically transmitted to the Raspberry Pi.
 
## LLM Server Dependency
This project depends on a local or remote LLM server (e.g., llm_server_monologg.py) that returns the following JSON format:
{
  "emotion": "happy",
  "response": "I'm glad to hear that!"
}
Ensure that the server is running and accessible at the address specified in config.py.

## Author
Developed by Yubin Wang
Department of Electrical, Electronic and Computer Engineering
University of Seoul