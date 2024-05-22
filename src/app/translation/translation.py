import os
import torch
import shutil

from transformers import pipeline
from transformers import TRANSFORMERS_CACHE

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

has_gpu = torch.cuda.is_available()
has_mps = "mps" if torch.backends.mps.is_available(
) and torch.backends.mps.is_built() else "cpu"
device = "mps" if has_mps else "cuda" if has_gpu else "cpu"

language_codes = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
}


def clean_up_when_close():
    shutil.rmtree(TRANSFORMERS_CACHE)


def translate_text(input: str, from_language: str, to_language: str) -> str:
    command = f"translation_{from_language}_to_{to_language}"
    prefix = f"translate {language_codes[from_language]} to {language_codes[to_language]}: "
    translation = pipeline(command, model="google-t5/t5-small", device=device)
    return translation(prefix+input)[0]["translation_text"]
