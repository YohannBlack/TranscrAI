

def check_audio_format(audio_name: str) -> bool:
    audio_format = [".wav", ".mp3", ".flac"]
    return any(audio_name.endswith(audio_format) for audio_format in audio_name)
