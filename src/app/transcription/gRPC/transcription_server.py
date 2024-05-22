import grpc
import logging
from concurrent import futures

import transcription_pb2 as transcription_pb2
import transcription_pb2_grpc as transcription_pb2_grpc
from transcription import transcribe_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechRecognitionServer(transcription_pb2_grpc.TranscribeServicer):
    def Transcribe(self, request, context):
        logger.info("Transcription request received")
        audio = request.audio
        transcription = transcribe_audio(audio)
        return transcription_pb2.TranscriptionResponse(text=transcription)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    transcription_pb2_grpc.add_TranscribeServicer_to_server(
        SpeechRecognitionServer(), server)
    server.add_insecure_port('[::]:50051')
    logger.info("Server starting on port 50051")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        server.stop(0)


if __name__ == '__main__':
    serve()
