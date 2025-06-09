# RadarX: Emotion-Aware AI Assistant using Radar and Camera (Ongoing)

## Overview

**RadarX** is an AI assistant that detects and responds to human emotions using a combination of **radar-based HRV (Heart Rate Variability)** sensing and **camera-based facial expression analysis**. This multimodal system is designed to support mental well-being, especially in socially isolated individuals, by providing emotionally adaptive feedback and conversation.

> This project addresses South Korea's rising mental health concerns by combining physiological and facial emotion recognition into a conversational AI system.

---

## Key Features

- **HRV Emotion Recognition (Radar)**
  - Extracts HRV parameters such as RMSSD, SDNN, LF, HF, meanHR, ampHR, etc.
  - Converts physiological signals to emotional metrics: *Valence* and *Arousal*

- **Facial Emotion Recognition (Camera)**
  - Detects face from camera feed and classifies emotion using DeepFace + MTCNN
  - Outputs emotion estimates (valence, arousal) and saves to JSON

- **AI Assistant Interaction**
  - Accepts speech input (PC)
  - Transcribes with Whisper, analyzes emotion (KoELECTRA), generates responses (pyttsx3)
  - Sends response audio to Raspberry Pi via SCP for playback

---
---

## System Architecture

### Radar-based HRV Emotion Detection
- **Hardware**: Radar sensor connected to Raspberry Pi
- **Pipeline**:
  1. Radar data → HRV signal extraction
  2. Feature extraction → Valence & Arousal estimation
  3. Data sent to PC via socket

### Camera-based Emotion Detection
- **Hardware**: Raspberry Pi camera
- **Pipeline**:
  1. Frame capture → Face detection (MTCNN)
  2. Emotion classification (DeepFace)
  3. Results stored as JSON with timestamp

### AI Assistant (PC-based)
- Speech-to-Text: `sounddevice + Whisper`
- Emotion Classification: `KoELECTRA`
- Text-to-Speech: `pyttsx3`
- Audio Transfer: `SCP → Raspberry Pi`
- Playback: `aplay`

---

## Demo Progress

- Radar HRV emotion recognition completed
- Facial emotion detection integrated and tested
- AI assistant speech interface running on PC
- Raspberry Pi receives and plays back emotion-aware response
- LCD + Camera integrated into 3D printed housing

---

## Timeline Summary

| Stage                          | Status   |
|-------------------------------|----------|
| Radar emotion system          | Done     |
| Camera emotion system         | Done     |
| AI voice assistant (PC-based) | Done     |
| LCD integration               | Done     |
| Full system demo              | Ongoing  |
| 3D printing & final packaging | Ongoing  |

---

## Contributors

- 왕유빈 
- 이진욱  
- 임현서