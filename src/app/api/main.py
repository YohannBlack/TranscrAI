import os
import grpc

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from tools.helper_functions import check_audio_format, split_audio
from gRPC_files.transcription.transcription_pb2 import TranscriptionRequest
from gRPC_files.transcription.transcription_pb2_grpc import TranscribeStub
from gRPC_files.translation.translation_pb2 import TranslationRequest
from gRPC_files.translation.translation_pb2_grpc import TranslateStub

app = Flask(__name__,
            template_folder="src/app/web/templates",)
CORS(app)

download_folder = "src/data"
app.config["DOWNLOAD_FOLDER"] = download_folder

os.makedirs(download_folder, exist_ok=True)

transcription_service_host = os.getenv("TRANSCRIPTION_SERVICE_HOST",
                                       "localhost")
transcription_channel = grpc.insecure_channel("localhost:50051",
                                              options=[('grpc.max_send_message_length', 100 * 1024 * 1024),
                                                       ('grpc.max_receive_message_length', 100 * 1024 * 1024)])
transcription_client = TranscribeStub(transcription_channel)

translatation_channel = grpc.insecure_channel("localhost:50052")
translation_client = TranslateStub(translatation_channel)


@app.route("/")
def index():
    return "Hello World"


@app.route("/api/v1/transcribe", methods=["POST"])
def transcribe():
    if 'audioFile' not in request.files:
        return jsonify({"error": "No file part in the request"}, 400)

    audio_file = request.files["audioFile"]

    if audio_file.filename == "":
        return jsonify({"error": "No file selected for uploading"}, 400)

    if not check_audio_format(audio_file.filename):
        return jsonify({"error": "Invalid file format, audio should be of format .wav, .mp3 or .flac"}, 400)

    is_long_audio = request.form['isLongAudio']

    if is_long_audio == "true":
        is_long_audio = True
    else:
        is_long_audio = False

    language = request.form['language']

    if audio_file:
        filename = audio_file.filename
        file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        audio_file.save(file_path)

        audio_chunks = split_audio(file_path)

    def generate_response():
        try:
            request_iterator = (TranscriptionRequest(
                audio=audio, language=language, isLongAudio=is_long_audio) for audio in audio_chunks)
            response_iterator = transcription_client.TranscribeStreaming(
                request_iterator)

            for response in response_iterator:
                if response.text is not None:
                    yield f"transcription: {response.text}\n\n".encode("utf-8")
        except Exception as e:
            return jsonify({"error": str(e)}, 500)

    if len(audio_chunks) != 1:
        return Response(stream_with_context(generate_response()), content_type="text/plain")
    else:
        audio = audio_chunks[0]
        transcription_request = TranscriptionRequest(
            audio=audio,
            language=language,
            isLongAudio=is_long_audio
        )
        response = transcription_client.Transcribe(transcription_request)

        return jsonify({
            "transcription": response.text,
        }, 200)


@app.route("/api/v1/translate", methods=["POST"])
def translate():
    if 'text' not in request.form:
        return jsonify({"error": "No text in the request"}, 400)

    text = request.form["text"]

    if text == "":
        return jsonify({"error": "No text in the request"}, 400)

    if 'from_language' not in request.form:
        return jsonify({"error": "No from_language in the request"}, 400)

    if 'to_language' not in request.form:
        return jsonify({"error": "No to_language in the request"}, 400)

    from_language = request.form["from_language"]
    to_language = request.form["to_language"]

    translation_request = TranslationRequest(
        input=text,
        from_language=from_language,
        to_language=to_language
    )
    response = translation_client.Translate(translation_request)

    return jsonify({
        "translation": response.text,
    }, 200)



if __name__ == "__main__":
    app.run(debug=True)
