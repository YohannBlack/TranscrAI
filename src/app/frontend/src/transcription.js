import React, { useState } from "react";
import "./styles/transcription.css";

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [transcription, setTranscription] = useState("");
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setAudioFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("audioFile", audioFile);
    formData.append("language", "english");
    formData.append("isLongAudio", "True");

    try {
      const response = await fetch("http://127.0.0.1:5000/api/v1/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const reader = response.body.getReader();
      let output = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        const transcriptionText = new TextDecoder("utf-8")
          .decode(value)
          .replace("transcription:", "");
        output += transcriptionText;

        setTranscription((prevTranscription) => prevTranscription + output);
      }
      setError(null); // Réinitialisez l'état d'erreur en cas de succès
    } catch (error) {
      setError(error.message);
      console.error("Error uploading file:", error);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    setAudioFile(file);
  };

  return (
    <div className="App">
      <header className="App-header">
        <p>Transcrire l'audio en texte</p>
        <div
          className="file-input-wrapper"
          onDragOver={handleDragOver}
          onDrop={handleDrop}>
          <input
            type="file"
            id="file"
            className="file-input"
            onChange={handleFileChange}
          />
          <label htmlFor="file" className="custom-file-upload">
            Selectionner un fichier ou deposer le fichier ici
          </label>
          {audioFile ? (
            <p className="file-name">{audioFile.name}</p>
          ) : (
            <p className="file-name">No file selected</p>
          )}
        </div>
        <button
          onClick={handleUpload}
          style={{ display: audioFile ? "inline-block" : "none" }}>
          Transcrire
        </button>

        {error && (
          <div className="error-message">
            <p>Error:</p>
            <p>{error}</p>
          </div>
        )}
        {transcription && (
          <div className="transcription-result">
            <p>{transcription}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
