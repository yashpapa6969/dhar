import torch
import torchaudio
import numpy as np
import pyaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
import noisereduce as nr
from scipy import signal
import queue
import threading
import time
import os
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder, preprocess_wav

import torch
import torchaudio
import numpy as np
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
import noisereduce as nr
from scipy import signal
import os
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder, preprocess_wav

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load environment variables
load_dotenv()

# Global configurations
SAMPLE_RATE = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Initialize models
model_id = "distil-whisper/distil-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, max_new_tokens=128, torch_dtype=torch_dtype, device=device)

diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HF_AUTH_TOKEN"))
voice_encoder = VoiceEncoder()

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_timestamps, _, read_audio, _, _) = utils

def preprocess_audio(audio):
    if SAMPLE_RATE != 16000:
        resampler = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)
        audio = resampler(torch.from_numpy(audio)).numpy()
    sos = signal.butter(10, 100, 'hp', fs=16000, output='sos')
    audio = signal.sosfilt(sos, audio)
    audio = nr.reduce_noise(y=audio, sr=16000)
    return audio

def perform_asr(audio):
    result = pipe(audio)
    return result["text"]

def perform_diarization(audio):
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    diarization = diarization_pipeline({"waveform": audio_tensor, "sample_rate": 16000})
    return diarization

def recognize_speaker(audio):
    preprocessed_wav = preprocess_wav(audio, source_sr=SAMPLE_RATE)
    embedding = voice_encoder.embed_utterance(preprocessed_wav)
    return embedding

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    audio_chunk = np.frombuffer(data, dtype=np.float32)

    # Preprocess audio
    audio = preprocess_audio(audio_chunk)

    # Perform VAD
    audio_tensor = torch.from_numpy(audio).float()
    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, threshold=0.5)

    if speech_timestamps:
        # Perform ASR
        transcription = perform_asr(audio)

        # Perform diarization
        diarization = perform_diarization(audio)

        # Recognize speaker
        speaker_embedding = recognize_speaker(audio)

        emit('transcription', {
            "transcription": transcription,
            "diarization": str(diarization),
            "speaker_embedding_shape": speaker_embedding.shape
        })

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)
