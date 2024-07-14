import grpc
import logging
from concurrent import futures

import gRPC.transcription_pb2 as transcription_pb2
import gRPC.transcription_pb2_grpc as transcription_pb2_grpc
from transcription import transcribe_audio, clean_up_when_close

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechRecognitionServer(transcription_pb2_grpc.TranscribeServicer):
    def Transcribe(self, request: transcription_pb2.TranscriptionRequest, context: grpc.ServicerContext) -> transcription_pb2.TranscriptionResponse:
        logger.info("Transcription request received")
        try:
            logger.info(
                f"Processing request: audio length: {len(request.audio)}, language: {request.language}")
            audio = request.audio
            language = request.language
            transcription = transcribe_audio(audio, language=language)
            logger.info("Transcription completed")
            return transcription_pb2.TranscriptionResponse(text=transcription)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            context.set_details(f"Exception processing request: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)

    def TranscribeStreaming(self, request_iterator, context: grpc.ServicerContext) -> transcription_pb2.TranscriptionResponse:
        logger.info("Transcription streaming request received")
        try:
            for request in request_iterator:
                logger.info(
                    f"Processing request: audio length: {len(request.audio)}, language: {request.language}")
                audio = request.audio
                language = request.language
                transcription = transcribe_audio(audio, language=language)
                logger.info("Transcription completed for a chunk")
                response = transcription_pb2.TranscriptionResponse(
                    text=transcription)
                print(response)
                yield response
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            context.set_details(f"Exception iterating requests: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_send_message_length', 100 * 1024 * 1024)     # 100 MB
        ]
    )
    transcription_pb2_grpc.add_TranscribeServicer_to_server(
        SpeechRecognitionServer(), server)
    server.add_insecure_port('[::]:50051')
    logger.info("Server starting on port 50051")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        # clean_up_when_close()
        server.stop(0)


if __name__ == '__main__':
    serve()
