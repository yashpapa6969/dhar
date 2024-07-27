from flask import Flask
from flask_socketio import SocketIO, emit
import logging
import numpy as np
import torch
import torchaudio
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from pyannote.audio import Pipeline
from noisereduce import reduce_noise
import os
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder, preprocess_wav
import asyncio
import threading
import json
import websockets

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Configuration for audio processing
SAMPLE_RATE = 16000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load models
model = AutoModelForSpeechSeq2Seq.from_pretrained("distil-whisper/distil-large-v3", torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True).to(DEVICE)
processor = AutoProcessor.from_pretrained("distil-whisper/distil-large-v3")
speech_recognition_pipeline = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, max_new_tokens=128, torch_dtype=TORCH_DTYPE, device=DEVICE)
diarization_pipeline = Pipeline.from_pretrained(model_id, use_auth_token=auth_token)
voice_encoder = VoiceEncoder()

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    process_audio_chunk(data)

def process_audio_chunk(data):
    try:
        if len(data) % 4 != 0:
            data = data[:-(len(data) % 4)]
        audio_np = np.frombuffer(data, dtype=np.float32)
        if audio_np.size == 0:
            logger.error("Empty audio buffer after alignment")
            return

        processed_audio = preprocess_audio(audio_np)
        if processed_audio is None:
            logger.error("Failed to preprocess audio")
            return

        # Additional processing logic
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")

async def websocket_audio_processor(websocket, path):
    global socketio
    try:
        while True:
            data = await websocket.recv()
            process_audio_chunk(data)
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")

def start_websocket_server():
    start_server = websockets.serve(websocket_audio_processor, "localhost", 8001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    # Start Flask-SocketIO server
    threading.Thread(target=lambda: socketio.run(app, host='0.0.0.0', port=5000, debug=True)).start()

    # Start WebSocket server
    threading.Thread(target=start_websocket_server).start()
