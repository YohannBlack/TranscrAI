import os
import grpc

from flask import Flask, render_template, request, jsonify

from gRPC_files.transcription_pb2 import TranscriptionRequest
from gRPC_files.transcription_pb2_grpc import TranscribeStub

app = Flask(__name__,
            template_folder="src/app/web/templates",)

download_folder = "src/data"
app.config["DOWNLOAD_FOLDER"] = download_folder

os.makedirs(download_folder, exist_ok=True)

transcription_service_host = os.getenv(
    "TRANSCRIPTION_SERVICE_HOST", "localhost")
transcription_channel = grpc.insecure_channel(
    f"localhost:50051")
transcription_client = TranscribeStub(transcription_channel)


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

    if audio_file:
        filename = audio_file.filename
        file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        audio_file.save(file_path)

        with open(file_path, "rb") as f:
            content = f.read()

        transcription_request = TranscriptionRequest(audio=content)

        response = transcription_client.Transcribe(transcription_request)

        return jsonify({
            "transcription": response.text,
        }, 200)


if __name__ == "__main__":
    app.run(debug=True)
