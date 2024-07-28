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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configurations
SAMPLE_RATE = 16000
CHUNK_DURATION = 30.0  # Process in 30-second chunks for better context
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_DURATION = 5.0  # 5-second overlap between chunks
OVERLAP_SIZE = int(SAMPLE_RATE * OVERLAP_DURATION)

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

# Load the onnx model
ort_session = ort.InferenceSession(vad_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
(get_speech_timestamps, _, read_audio, *_) = utils

# Transcription handling
transcription_buffer = deque(maxlen=10)  # Store last 10 transcriptions
QUERY_END_PHRASES = [".", "?", "!"]
SILENCE_THRESHOLD = 2.0  # seconds of silence to consider end of query

def preprocess_audio(audio):
    audio_tensor = torch.from_numpy(audio).float()
    audio_tensor = torchaudio.functional.resample(audio_tensor, SAMPLE_RATE, 16000)
    audio_tensor = torchaudio.functional.highpass_biquad(audio_tensor, 16000, 100)
    audio = nr.reduce_noise(y=audio_tensor.numpy(), sr=16000)
    return audio

def perform_asr(audio):
    result = pipe(audio)
    return result["text"]

def perform_diarization(audio):
    audio_tensor = torch.from_numpy(audio).float().to(device)
    
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # Adds a channel dimension [1, time]
    diarization = diarization_pipeline({"waveform": audio_tensor, "sample_rate": 16000})
    return diarization

async def process_audio_chunk(audio):
    try:
        audio1 = torch.from_numpy(audio).float()
        speech_timestamps = get_speech_timestamps(audio1, vad_model, threshold=0.5)
        if not speech_timestamps:
            return None, None, None

        audio = preprocess_audio(audio)
        transcription = perform_asr(audio)
        diarization = perform_diarization(audio)
        speaker_embedding = voice_encoder.embed_utterance(preprocess_wav(audio))
        return transcription, diarization, speaker_embedding
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        return None, None, None

def is_query_complete(transcription):
    return any(transcription.strip().endswith(phrase) for phrase in QUERY_END_PHRASES)

async def send_transcription(websocket, transcription, is_final=False):
    await websocket.send(json.dumps({
        'type': 'final' if is_final else 'interim',
        'text': transcription
    }))
    logger.info(f"{'Final' if is_final else 'Interim'} Transcription: {transcription}")

async def handle_client(websocket, path):
    logger.info("Client connected")
    audio_buffer = deque(maxlen=CHUNK_SIZE + OVERLAP_SIZE)
    last_speech_time = asyncio.get_event_loop().time()

    async for message in websocket:
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
        audio_buffer.extend(audio_np)

        # Process chunks
        if len(audio_buffer) >= CHUNK_SIZE:
            chunk = np.array(list(audio_buffer)[-CHUNK_SIZE:])
            
            transcription, diarization, speaker_embedding = await process_audio_chunk(chunk)

            if transcription:
                last_speech_time = asyncio.get_event_loop().time()
                transcription_buffer.append(transcription)
                
                # Combine recent transcriptions for context
                combined_transcription = " ".join(transcription_buffer)
                
                if is_query_complete(combined_transcription):
                    await send_transcription(websocket, combined_transcription, is_final=True)
                    transcription_buffer.clear()
                else:
                    await send_transcription(websocket, combined_transcription, is_final=False)
            else:
                # Check if silence duration exceeds the threshold
                current_time = asyncio.get_event_loop().time()
                if current_time - last_speech_time > SILENCE_THRESHOLD and transcription_buffer:
                    combined_transcription = " ".join(transcription_buffer)
                    await send_transcription(websocket, combined_transcription, is_final=True)
                    transcription_buffer.clear()

            if diarization:
                logger.info(f"Diarization: {diarization}")
            if speaker_embedding is not None:
                logger.info(f"Speaker embedding shape: {speaker_embedding.shape}")

            # Keep the overlap for the next chunk
            audio_buffer = deque(list(audio_buffer)[-OVERLAP_SIZE:], maxlen=CHUNK_SIZE + OVERLAP_SIZE)

async def main():
    server = await websockets.serve(handle_client, "localhost", 8001)
    logger.info("Server started. Press Ctrl+C to stop the server.")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
