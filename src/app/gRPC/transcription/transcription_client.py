import logging

import grpc
from . import transcription_pb2
from . import transcription_pb2_grpc


class TranscriptionClient:
    def __init__(self):
        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = transcription_pb2_grpc.TranscriptionServiceStub(
            self.channel)

    def transcribe(self, audio_path: str, is_long_audio: bool, language_code: str = "en-US"):
        request = transcription_pb2.TranscribeRequest(
            audio=audio_path, is_long_audio=is_long_audio, language=language_code)
        response = self.stub.Transcribe(request)
        return response.transcript, response.confidence
