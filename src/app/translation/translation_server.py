import grpc
import logging
from concurrent import futures

import gRPC.translation_pb2 as translation_pb2
import gRPC.translation_pb2_grpc as translation_pb2_grpc
from translation import translate_text, clean_up_when_close

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationServer(translation_pb2_grpc.TranslateServicer):
    def Translate(self, request, context):
        logger.info("Translation request received")
        text = request.input
        from_language = request.from_language
        to_language = request.to_language
        translation = translate_text(input_text=text,
                                     from_language=from_language,
                                     to_language=to_language)
        logger.info(f"Translation: {translation}")
        return translation_pb2.TranslationResponse(text=translation)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    translation_pb2_grpc.add_TranslateServicer_to_server(
        TranslationServer(), server)
    server.add_insecure_port('[::]:50052')
    logger.info("Server starting on port 50052")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        clean_up_when_close()
        server.stop(0)


if __name__ == '__main__':
    serve()
