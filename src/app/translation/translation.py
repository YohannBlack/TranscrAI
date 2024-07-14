import os
import torch
import shutil

from transformers import pipeline, M2M100Tokenizer, M2M100ForConditionalGeneration
from transformers import TRANSFORMERS_CACHE

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

has_gpu = torch.cuda.is_available()
has_mps = "mps" if torch.backends.mps.is_available(
) and torch.backends.mps.is_built() else "cpu"
device = "mps" if has_mps else "cuda" if has_gpu else "cpu"

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

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


def translate_text(input_text: str, from_language: str, to_language: str, model=model, tokenizer=tokenizer) -> str:

    if from_language == to_language:
        return input_text

    tokenizer.src_lang = from_language
    encoded_text = tokenizer(input_text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text,
                                      forced_bos_token_id=tokenizer.get_lang_id(to_language))
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
