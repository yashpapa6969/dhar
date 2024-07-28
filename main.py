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
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

if torch.cuda.is_available():
    diarization_pipeline.to(torch.device("cuda"))

# Initialize speaker recognition model
voice_encoder = VoiceEncoder()

# Initialize VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True)
(get_speech_timestamps, _, read_audio, _, _) = utils

client_websocket = None
buffer = np.array([])


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
    audio1= torch.from_numpy(audio).float().to(device)
    diarization = diarization_pipeline({"waveform": audio1, "sample_rate": 16000})
    return diarization


def process_audio_chunk(audio):
    try:
        audio1= torch.from_numpy(audio).float()
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


async def send_to_client(message):
    if client_websocket:
        await client_websocket.send(message)


async def handle_client(websocket, path):
    global client_websocket, buffer
    logger.info("Client connected")
    client_websocket = websocket

    async for message in websocket:
        # Extract metadata
        metadata_length = int.from_bytes(message[:4], byteorder='little')
        metadata_json = message[4:4 + metadata_length].decode('utf-8')
        metadata = json.loads(metadata_json)
        sample_rate = metadata['sampleRate']

        # Extract audio data
        audio_data = message[4 + metadata_length:]

        # Convert 16-bit PCM to 32-bit float
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample if necessary
        if sample_rate != SAMPLE_RATE:
            audio_np = signal.resample(audio_np, int(len(audio_np) * SAMPLE_RATE / sample_rate))

        # Add to buffer
        buffer = np.concatenate((buffer, audio_np))

        # Process in 2-second chunks
        chunk_size = 2 * SAMPLE_RATE
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]

            # Process the chunk
            transcription, diarization, speaker_embedding = process_audio_chunk(chunk)

            if transcription:
                # Send realtime transcription
                await send_to_client(json.dumps({
                    'type': 'realtime',
                    'text': transcription
                }))

                # Log results
                logger.info(f"Transcription: {transcription}")
                logger.info(f"Diarization: {diarization}")
                logger.info(f"Speaker embedding shape: {speaker_embedding.shape}")

                # You can add logic here to determine when to send a 'fullSentence' message


async def main():
    server = await websockets.serve(handle_client, "0.0.0.0", 5000)
    logger.info("Server started. Press Ctrl+C to stop the server.")
    await asyncio.Future() 


if __name__ == "__main__":
    print("Starting server, please wait...")
    asyncio.run(main())
