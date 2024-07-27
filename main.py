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

# Load environment variables
load_dotenv()

# Global configurations
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paFloat32

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
diarization_pipeline = Pipeline.from_pretrained( "pyannote/speaker-diarization-3.1", 
                                                use_auth_token="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")

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

def perform_asr(audio):
    result = pipe(audio)
    return result["text"]

def perform_diarization(audio):
    diarization = diarization_pipeline({"waveform": torch.from_numpy(audio), "sample_rate": 16000})
    return diarization

def recognize_speaker(audio):
    preprocessed_wav = preprocess_wav(audio)
    embedding = voice_encoder.embed_utterance(preprocessed_wav)
    return embedding

def process_audio():
    buffer = np.array([])
    
    while True:
        # Get audio chunk from queue
        chunk = audio_queue.get()
        if chunk is None:
            break
        
        # Add chunk to buffer
        buffer = np.concatenate((buffer, chunk))
        
        # Process if buffer is large enough
        if len(buffer) >= SAMPLE_RATE * 2:  # Process 2 seconds of audio
            audio = buffer[:SAMPLE_RATE * 2]
            buffer = buffer[SAMPLE_RATE * 2:]
            
            # Preprocess audio
            audio = preprocess_audio(audio)
            
            # Perform VAD
            speech_timestamps = get_speech_timestamps(torch.from_numpy(audio), vad_model, threshold=0.5)
            
            if speech_timestamps:
                # Perform ASR
                transcription = perform_asr(audio)
                
                # Perform diarization
                diarization = perform_diarization(audio)
                
                # Recognize speaker
                speaker_embedding = recognize_speaker(audio)
                
                print(f"Transcription: {transcription}")
                print(f"Diarization: {diarization}")
                print(f"Speaker embedding shape: {speaker_embedding.shape}")

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

# Keep the stream active
try:
    while stream.is_active():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_queue.put(None)
    processing_thread.join()
