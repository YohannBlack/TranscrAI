import torch
from transformers import pipeline

has_gpu = torch.cuda.is_available()
has_mps = "mps" if torch.backends.mps.is_available() \
    and torch.backends.mps.is_built() else "cpu"
device = "mps" if has_mps else "cuda" if has_gpu else "cpu"
pipe = pipeline("automatic-speech-recognition",
                model="openai/whisper-base", device="cpu")


def check_audio_format(audio_path: str) -> bool:
    audio_formats = [".wav", ".mp3", ".flac"]
    return any(audio_path.endswith(audio_format) for audio_format in audio_formats)


def transcribe_audio(audio_path: str, is_long_audio: bool = False, language: str = "english") -> str:

    if is_long_audio:
        return pipe(
            audio_path,
            max_new_tokens=256,
            generate_kwargs={"task": "transcribe",
                             "language": language},
            chunk_length_s=30,
            batch_size=8
        )["text"]
    else:
        return pipe(
            audio_path,
            max_new_tokens=256,
            generate_kwargs={"task": "transcribe",
                             "language": language}
        )["text"]
