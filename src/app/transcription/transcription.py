import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import shutil
from transformers import pipeline
from transformers import TRANSFORMERS_CACHE

has_gpu = torch.cuda.is_available()
has_mps = "mps" if torch.backends.mps.is_available() \
    and torch.backends.mps.is_built() else None
device = "mps" if has_mps else "cuda" if has_gpu else "cpu"
pipe = pipeline("automatic-speech-recognition",
                model="openai/whisper-base", device=device)


def check_audio_format(audio_path: str) -> bool:
    audio_formats = [".wav", ".mp3", ".flac"]
    return any(audio_path.endswith(audio_format) for audio_format in audio_formats)


def clean_up_when_close():
    print(TRANSFORMERS_CACHE)
    shutil.rmtree(TRANSFORMERS_CACHE)


def transcribe_audio(audio_path: bytes, language: str = "english") -> str:

        return pipe(
            audio_path,
            max_new_tokens=256,
            generate_kwargs={"task": "transcribe",
                             "language": language},
            # chunk_length_s=30,
            # batch_size=8
        )["text"]
