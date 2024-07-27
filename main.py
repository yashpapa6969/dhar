import torch
import torchaudio
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
import noisereduce as nr
from resemblyzer import VoiceEncoder, preprocess_wav
import logging
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global configurations
SAMPLE_RATE = 16000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Initialize Distil-Whisper model for ASR
model_id = "distil-whisper/distil-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

# Initialize speaker diarization pipeline
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                use_auth_token="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")

# Move diarization pipeline to GPU if available
if torch.cuda.is_available():
    diarization_pipeline.to(torch.device("cuda"))

# Initialize speaker recognition model
voice_encoder = VoiceEncoder()

# Initialize VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True)
(get_speech_timestamps, _, read_audio, _, _) = utils

def preprocess_audio(audio_np):
    """
    Preprocesses the audio data received as a NumPy array.
    
    :param audio_np: A NumPy array containing the audio data.
    :return: A NumPy array with processed audio.
    """
    # Check that input is a NumPy array
    if not isinstance(audio_np, np.ndarray):
        logger.error("Expected audio input to be a NumPy array.")
        return None

    # Convert to tensor and ensure it's floating point
    audio_tensor = torch.from_numpy(audio_np).float()

    # Check and handle non-finite values (NaN, Inf)
    if not torch.isfinite(audio_tensor).all():
        logger.error("Audio buffer contains non-finite values. Replacing them with zeros.")
        audio_tensor = torch.where(torch.isfinite(audio_tensor), audio_tensor, torch.zeros_like(audio_tensor))

    # Resample and apply high-pass filter
    audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=SAMPLE_RATE, new_freq=16000)
    audio_tensor = torchaudio.functional.highpass_biquad(audio_tensor, sample_rate=16000, cutoff_freq=100)

    # Noise reduction using 'noisereduce' which operates on NumPy arrays
    # Convert tensor back to numpy array for noise reduction
    processed_audio = audio_tensor.numpy()
    processed_audio = nr.reduce_noise(y=processed_audio, sr=16000)

    return processed_audio

def perform_asr(audio):
    result = pipe(audio)
    return result["text"]

def perform_diarization(audio):
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    diarization = diarization_pipeline({"waveform": audio_tensor, "sample_rate": 16000})
    return diarization

def recognize_speaker(audio):
    preprocessed_wav = preprocess_wav(audio)
    embedding = voice_encoder.embed_utterance(preprocessed_wav)
    return embedding

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    try:
        logger.info("Received audio chunk")
        if len(data) % 4 != 0:
            data = data[:-(len(data) % 4)]
            logger.info(f"Truncated audio data to {len(data)} bytes for alignment")

        audio = np.frombuffer(data, dtype=np.float32)
        logger.info(f"Audio buffer size: {audio.size}")

        if audio.size == 0:
            logger.error("Empty audio buffer after alignment")
            return

        audio = preprocess_audio(audio)
        if audio is None:
            logger.error("Failed to preprocess audio")
            return

        # Further processing...
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")




if __name__ == "__main__":
    logger.info("Starting server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
