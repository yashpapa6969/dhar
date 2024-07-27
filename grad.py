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
import gradio as gr
import logging
from collections import deque
import argparse
import socket

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Real-time ASR with speaker diarization")
parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
parser.add_argument("--chunk_size", type=int, default=1024, help="Audio chunk size")
parser.add_argument("--buffer_seconds", type=float, default=2.0, help="Audio buffer size in seconds")
parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio interface on")
parser.add_argument("--public", action="store_true", help="Make the interface publicly available")
args = parser.parse_args()

# Global configurations
SAMPLE_RATE = args.sample_rate
CHUNK_SIZE = args.chunk_size
CHANNELS = 1
FORMAT = pyaudio.paFloat32
BUFFER_SECONDS = args.buffer_seconds

# Initialize CUDA if available
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
                                                use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

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

# Initialize queue for audio chunks
audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(np.frombuffer(in_data, dtype=np.float32))
    return (None, pyaudio.paContinue)

def preprocess_audio(audio):
    try:
        # Resample
        if SAMPLE_RATE != 16000:
            resampler = torchaudio.transforms.Resample(SAMPLE_RATE, 16000)
            audio = resampler(torch.from_numpy(audio)).numpy()

        # Apply high-pass filter
        sos = signal.butter(10, 100, 'hp', fs=16000, output='sos')
        audio = signal.sosfilt(sos, audio)

        # Noise reduction
        audio = nr.reduce_noise(y=audio, sr=16000)

        return audio
    except Exception as e:
        logging.error(f"Error in preprocess_audio: {e}")
        return None

def perform_asr(audio):
    try:
        result = pipe(audio)
        return result["text"]
    except Exception as e:
        logging.error(f"Error in perform_asr: {e}")
        return ""

def perform_diarization(audio):
    try:
        diarization = diarization_pipeline({"waveform": torch.from_numpy(audio).to(device), "sample_rate": 16000})
        return diarization
    except Exception as e:
        logging.error(f"Error in perform_diarization: {e}")
        return None

def recognize_speaker(audio):
    try:
        preprocessed_wav = preprocess_wav(audio)
        embedding = voice_encoder.embed_utterance(preprocessed_wav)
        return embedding
    except Exception as e:
        logging.error(f"Error in recognize_speaker: {e}")
        return None

def process_audio():
    buffer = deque(maxlen=int(SAMPLE_RATE * BUFFER_SECONDS))

    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break

        buffer.extend(chunk)

        if len(buffer) == buffer.maxlen:
            audio = np.array(buffer)

            audio = preprocess_audio(audio)
            if audio is None:
                continue

            speech_timestamps = get_speech_timestamps(torch.from_numpy(audio), vad_model, threshold=0.5)

            if speech_timestamps:
                transcription = perform_asr(audio)
                diarization = perform_diarization(audio)
                speaker_embedding = recognize_speaker(audio)

                logging.info(f"Transcription: {transcription}")
                logging.info(f"Diarization: {diarization}")
                if speaker_embedding is not None:
                    logging.info(f"Speaker embedding shape: {speaker_embedding.shape}")

                yield transcription

# Gradio interface
def transcribe_audio(audio):
    audio = audio.squeeze()  # Remove channel dimension if present
    audio = (audio * 32768).astype(np.int16)  # Convert to int16
    audio = audio.astype(np.float32) / 32768.0  # Convert back to float32
    
    audio = preprocess_audio(audio)
    if audio is None:
        return "Error preprocessing audio"

    transcription = perform_asr(audio)
    return transcription

iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(source="microphone", streaming=True),
    outputs="text",
    live=True,
    title="Real-time ASR",
    description="Speak into your microphone to see real-time transcription."
)

# Start processing thread
processing_thread = threading.Thread(target=process_audio)
processing_thread.start()

# Initialize PyAudio
p = pyaudio.PyAudio()

# Start audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=audio_callback)

stream.start_stream()

# Launch Gradio interface
iface.launch(
    server_name="0.0.0.0" if args.public else None,
    server_port=args.port,
    share=args.public
)

if args.public:
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Public URL: http://{ip_address}:{args.port}")

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
audio_queue.put(None)
processing_thread.join()
