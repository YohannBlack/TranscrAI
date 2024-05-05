import os
import grpc

from flask import Flask, render_template, request, jsonify

from src.app.gRPC.transcription.transcription_pb2 import TranscribeRequest
from src.app.gRPC.transcription.transcription_pb2_grpc import TranscriptionServiceStub

app = Flask(__name__,
            template_folder="src/app/web/templates",)

transcription_service_host = os.getenv(
    "TRANSCRIPTION_SERVICE_HOST", "localhost")
transcription_channel = grpc.insecure_channel(
    f"{transcription_service_host}:50051")
transcription_client = TranscriptionServiceStub(transcription_channel)


@app.route("/")
def index():
    return render_template("transcription.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files["audioFile"]
    language = request.form["language"]
    is_long_audio = request.form.get("isLong", False)

    transcription_request = TranscribeRequest(audio=audio_file,
                                              language=language,
                                              is_long_audio=is_long_audio)

    response = transcription_client.Transcribe(transcription_request)

    return jsonify({
        "transcription": response.transcript,
    }, 200)


if __name__ == "__main__":
    app.run(debug=True)
