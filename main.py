import asyncio
import websockets
import json
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
import noisereduce as nr
from resemblyzer import VoiceEncoder, preprocess_wav
import logging
from collections import deque
import onnxruntime as ort
import re
from dataclasses import dataclass
from typing import List, Tuple

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
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Initialize speaker diarization pipeline
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                use_auth_token="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")

if torch.cuda.is_available():
    diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))

# Initialize speaker recognition model
voice_encoder = VoiceEncoder()

# Initialize Silero VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True,
                                  onnx=True)

(get_speech_timestamps, _, read_audio, *_) = utils

@dataclass
class TranscriptionSegment:
    text: str
    start_time: float
    end_time: float

class AudioBuffer:
    def __init__(self, max_duration: float = 60.0):
        self.buffer = deque(maxlen=int(max_duration * SAMPLE_RATE))

    def add(self, audio: np.ndarray):
        self.buffer.extend(audio)

    def get(self, duration: float) -> np.ndarray:
        size = int(duration * SAMPLE_RATE)
        return np.array(list(self.buffer)[-size:])

class TranscriptionManager:
    def __init__(self):
        self.segments: List[TranscriptionSegment] = []
        self.last_finalized_time: float = 0

    def add_segment(self, text: str, start_time: float, end_time: float):
        self.segments.append(TranscriptionSegment(text, start_time, end_time))
        self.segments.sort(key=lambda x: x.start_time)

    def get_concatenated(self, current_time: float) -> Tuple[str, bool]:
        result = []
        is_final = False
        for segment in self.segments:
            if segment.end_time <= current_time - 2.0:  # Consider segments final after 2 seconds
                result.append(segment.text)
                self.last_finalized_time = segment.end_time
                is_final = True
            else:
                result.append(segment.text)
        return " ".join(result), is_final

    def clear_finalized(self):
        self.segments = [seg for seg in self.segments if seg.end_time > self.last_finalized_time]

# Global state
audio_buffer = AudioBuffer(max_duration=60.0)  # 60 seconds buffer
transcription_manager = TranscriptionManager()

QUERY_END_PHRASES = [".", "?", "!"]
SILENCE_THRESHOLD = 1.0  # seconds of silence to consider end of query
MIN_SPEECH_DURATION = 0.3  # seconds

def preprocess_audio(audio):
    audio_tensor = torch.from_numpy(audio).float()
    audio_tensor = torchaudio.functional.resample(audio_tensor, SAMPLE_RATE, 16000)
    audio_tensor = torchaudio.functional.highpass_biquad(audio_tensor, 16000, 100)
    audio = nr.reduce_noise(y=audio_tensor.numpy(), sr=16000)
    return audio

def perform_asr(audio):
    result = pipe(audio)
    return result

def perform_diarization(audio):
    audio_tensor = torch.from_numpy(audio).float().to(device)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Adds a channel dimension [1, time]
    diarization = diarization_pipeline({"waveform": audio_tensor, "sample_rate": 16000})
    return diarization

async def process_audio_chunk(audio, chunk_start_time):
    try:
        audio_float = torch.from_numpy(audio).float()
        speech_timestamps = get_speech_timestamps(audio_float, vad_model, threshold=0.5, min_speech_duration_ms=MIN_SPEECH_DURATION * 1000)
        if not speech_timestamps:
            return None, None, None

        audio = preprocess_audio(audio)
        asr_result = perform_asr(audio)
        diarization = perform_diarization(audio)
        speaker_embedding = voice_encoder.embed_utterance(preprocess_wav(audio))

        for chunk in asr_result["chunks"]:
            transcription_manager.add_segment(
                chunk["text"],
                chunk_start_time + chunk["timestamp"][0],
                chunk_start_time + chunk["timestamp"][1]
            )

        return asr_result["text"], diarization, speaker_embedding
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        return None, None, None

async def send_transcription(websocket, text: str, is_final: bool):
    await websocket.send(json.dumps({
        'type': 'final' if is_final else 'interim',
        'text': text
    }))

async def handle_client(websocket, path):
    logger.info("Client connected")
    chunk_start_time = 0

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
                audio_np = torchaudio.functional.resample(torch.from_numpy(audio_np), sample_rate, SAMPLE_RATE).numpy()

            # Add to buffer
            audio_buffer.add(audio_np)

            # Process chunks
            chunk = audio_buffer.get(CHUNK_DURATION)
            transcription, diarization, speaker_embedding = await process_audio_chunk(chunk, chunk_start_time)

            if transcription:
                concatenated_transcription, is_final = transcription_manager.get_concatenated(chunk_start_time + CHUNK_DURATION)
                await send_transcription(websocket, concatenated_transcription, is_final)

                if is_final:
                    transcription_manager.clear_finalized()

            if diarization:
                logger.info(f"Diarization: {diarization}")
            if speaker_embedding is not None:
                logger.info(f"Speaker embedding shape: {speaker_embedding.shape}")

            chunk_start_time += CHUNK_DURATION

        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")

async def main():
    server = await websockets.serve(handle_client, "0.0.0.0", 5000)
    logger.info("Server started. Press Ctrl+C to stop the server.")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
