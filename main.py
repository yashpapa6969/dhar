import os
import asyncio
import threading
import numpy as np
import torch
import torchaudio
from flask import Flask
from flask_socketio import SocketIO
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from resemblyzer import VoiceEncoder, preprocess_wav
from dotenv import load_dotenv
import websockets
from websockets import serve
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration for audio processing
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load models
model = AutoModelForSpeechSeq2Seq.from_pretrained("distil-whisper/distil-large-v3", torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True).to(DEVICE)
processor = AutoProcessor.from_pretrained("distil-whisper/distil-large-v3")
speech_recognition_pipeline = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, max_new_tokens=128, torch_dtype=TORCH_DTYPE, device=DEVICE)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
voice_encoder = VoiceEncoder()

def preprocess_audio(audio_np):
    """
    Preprocess the audio data received as a NumPy array.
    
    :param audio_np: A NumPy array containing the audio data.
    :return: A NumPy array with processed audio.
    """
    # Convert to tensor and ensure it's floating point
    audio_tensor = torch.from_numpy(audio_np).float()
    
    # Resample and apply high-pass filter
    audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=SAMPLE_RATE, new_freq=16000)
    audio_tensor = torchaudio.functional.highpass_biquad(audio_tensor, sample_rate=16000, cutoff_freq=100)

    # Convert tensor back to numpy array
    processed_audio = audio_tensor.numpy()

    return processed_audio

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    try:
        audio_np = np.frombuffer(data, dtype=np.float32)
        if audio_np.size == 0:
            logger.error("Empty audio buffer")
            return

        processed_audio = preprocess_audio(audio_np)
        # Further processing logic here
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")

def flask_thread():
    # Run Flask without debug and reloader
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)

async def websocket_audio_processor(websocket, path):
    try:
        while True:
            data = await websocket.recv()
            handle_audio_chunk(data)
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")

def websocket_server():
    # Create a new event loop for the thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start WebSocket server
    start_server = serve(websocket_audio_processor, "localhost", 8001)

    loop.run_until_complete(start_server)
    loop.run_forever()

if __name__ == "__main__":
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=flask_thread)
    flask_thread.start()

    # Start WebSocket server in a separate thread
    ws_thread = threading.Thread(target=websocket_server)
    ws_thread.start()
