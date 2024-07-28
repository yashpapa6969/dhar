import asyncio
import websockets
import json
import numpy as np
from scipy import signal
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
import noisereduce as nr
from resemblyzer import VoiceEncoder, preprocess_wav
import logging
from collections import deque
from dataclasses import dataclass
from threading import Lock
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configurations
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # Process in 0.5-second chunks for lower latency
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Initialize CUDA if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@dataclass
class AudioChunk:
    audio: np.ndarray
    timestamp: float

class AudioBuffer:
    def __init__(self, max_duration: float = 30.0):
        self.buffer = deque()
        self.max_size = int(max_duration * SAMPLE_RATE)
        self.lock = Lock()

    def add(self, chunk: np.ndarray):
        with self.lock:
            self.buffer.extend(chunk)
            while len(self.buffer) > self.max_size:
                self.buffer.popleft()

    def get(self, duration: float) -> np.ndarray:
        size = int(duration * SAMPLE_RATE)
        with self.lock:
            return np.array(list(self.buffer)[-size:])

class TranscriptionManager:
    def __init__(self):
        self.transcriptions = []
        self.lock = Lock()

    def add(self, text: str, start_time: float, end_time: float):
        with self.lock:
            self.transcriptions.append((text, start_time, end_time))

    def get_concatenated(self) -> str:
        with self.lock:
            return " ".join([t[0] for t in sorted(self.transcriptions, key=lambda x: x[1])])

    def clear(self):
        with self.lock:
            self.transcriptions.clear()

# Initialize models
logger.info("Initializing models...")

# ASR Model (using the provided Distil-Whisper model)
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
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Speaker Diarization
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                use_auth_token="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")
if torch.cuda.is_available():
    diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))

# Voice Activity Detection
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                  onnx=True)
get_speech_timestamps, _, read_audio, _, _ = utils

# Speaker Recognition
voice_encoder = VoiceEncoder()

# Global state
audio_buffer = AudioBuffer(max_duration=60.0)  # 60 seconds buffer
transcription_manager = TranscriptionManager()

QUERY_END_PHRASES = [".", "?", "!"]
SILENCE_THRESHOLD = 1.5  # seconds
MIN_SPEECH_DURATION = 0.3  # seconds

def preprocess_audio(audio: np.ndarray) -> np.ndarray:
    audio_tensor = torch.from_numpy(audio).float()
    audio_tensor = torchaudio.functional.resample(audio_tensor, SAMPLE_RATE, 16000)
    audio_tensor = torchaudio.functional.highpass_biquad(audio_tensor, 16000, 100)
    audio = nr.reduce_noise(y=audio_tensor.numpy(), sr=16000)
    return audio

def perform_asr(audio: np.ndarray) -> dict:
    return pipe(audio, return_timestamps=True)

def perform_diarization(audio: np.ndarray) -> dict:
    audio_tensor = torch.from_numpy(audio).float().to(device)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    return diarization_pipeline({"waveform": audio_tensor, "sample_rate": SAMPLE_RATE})

async def process_audio_chunk(chunk: AudioChunk):
    try:
        audio1 = torch.from_numpy(chunk.audio).float()
        speech_timestamps = get_speech_timestamps(audio1, vad_model, threshold=0.5)
        
        if not speech_timestamps:
            return None, None, None, None

        audio = preprocess_audio(chunk.audio)
        asr_result = perform_asr(audio)
        diarization = perform_diarization(audio)
        speaker_embedding = voice_encoder.embed_utterance(preprocess_wav(audio))
        
        return asr_result["text"], asr_result["chunks"], diarization, speaker_embedding
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        return None, None, None, None

async def send_transcription(websocket, text: str, is_final: bool):
    await websocket.send(json.dumps({
        'type': 'final' if is_final else 'interim',
        'text': text
    }))

async def handle_client(websocket, path):
    logger.info("Client connected")
    last_speech_time = time.time()
    speech_active = False

    async for message in websocket:
        try:
            # Extract metadata and audio data
            metadata_length = int.from_bytes(message[:4], byteorder='little')
            metadata_json = message[4:4+metadata_length].decode('utf-8')
            metadata = json.loads(metadata_json)
            sample_rate = metadata['sampleRate']
            audio_data = message[4+metadata_length:]
            
            # Convert 16-bit PCM to 32-bit float
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample if necessary
            if sample_rate != SAMPLE_RATE:
                audio_np = signal.resample(audio_np, int(len(audio_np) * SAMPLE_RATE / sample_rate))

            # Add to buffer
            timestamp = time.time()
            audio_buffer.add(AudioChunk(audio_np, timestamp))

            # Process chunks
            chunk = audio_buffer.get(CHUNK_DURATION)
            transcription, timestamps, diarization, speaker_embedding = await process_audio_chunk(AudioChunk(chunk, timestamp))

            if transcription:
                last_speech_time = time.time()
                speech_active = True
                
                # Add transcription with timestamps
                for t in timestamps:
                    transcription_manager.add(t['text'], t['timestamp'][0] + timestamp, t['timestamp'][1] + timestamp)
                
                # Send interim transcription
                concatenated_transcription = transcription_manager.get_concatenated()
                await send_transcription(websocket, concatenated_transcription, False)
                
                # Check for query end
                if any(concatenated_transcription.strip().endswith(phrase) for phrase in QUERY_END_PHRASES):
                    await send_transcription(websocket, concatenated_transcription, True)
                    transcription_manager.clear()
                    speech_active = False
            else:
                # Check for silence
                current_time = time.time()
                if speech_active and current_time - last_speech_time > SILENCE_THRESHOLD:
                    concatenated_transcription = transcription_manager.get_concatenated()
                    if concatenated_transcription:
                        await send_transcription(websocket, concatenated_transcription, True)
                        transcription_manager.clear()
                    speech_active = False

            if diarization:
                logger.info(f"Diarization: {diarization}")
            if speaker_embedding is not None:
                logger.info(f"Speaker embedding shape: {speaker_embedding.shape}")

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")

async def main():
    server = await websockets.serve(handle_client, "0.0.0.0", 5000)
    logger.info("Server started. Press Ctrl+C to stop the server.")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
