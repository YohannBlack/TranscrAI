from typing import List
from pydub import AudioSegment

def check_audio_format(audio_name: str) -> bool:
    audio_format = [".wav", ".mp3", ".flac"]
    return any(audio_name.endswith(audio_format) for audio_format in audio_name)


def split_audio(file_path: str) -> List:
    audio = AudioSegment.from_file(file_path)

    duration_ms = len(audio)

    if duration_ms > 5 * 60 * 1000:
        num_chunks = int(duration_ms / (30 * 1000)) + 1

        chunks = []

        for i in range(num_chunks):
            start = i * 30 * 1000
            end = (i + 1) * 30 * 1000

            if end > duration_ms:
                end = duration_ms

            chunk = audio[start:end]

            chunk_bytes = chunk.export(format="wav", codec="pcm_s16le").read()

            chunks.append(chunk_bytes)

        return chunks

    audio_content = audio.export(format="wav", codec="pcm_s16le").read()
    return [audio_content]
