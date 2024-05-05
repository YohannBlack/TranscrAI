import logging
from concurrent import futures

import grpc
from . import transcription_pb2
from . import transcription_pb2_grpc
from ...transcription.transcription import *


class TranscriptionService(transcription_pb2_grpc.TranscriptionServiceServicer):
    def Transcribe(self, request, context):
        logging.info(f"Transcribe request: {request}")

        # transcription
        transcription = transcribe_audio(request.audio,
                                         request.is_long_audio,
                                         request.language)
        return transcription_pb2.TranscribeResponse(transcript=transcription)

    def serve():
        port = "50051"
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(
            TranscriptionService(), server)
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        logging.info(f"Server started on port {port}")
        server.wait_for_termination()
