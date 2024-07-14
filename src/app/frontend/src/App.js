import React, { useState } from 'react';
import './App.css';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [transcription, setTranscription] = useState("");
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setAudioFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!audioFile) {
      setError('Please select an audio file first');
      return;
    }

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

      const data = await response.json();
      setTranscription(data[0].transcription);
      setError(null); // Reset error state on success
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
        <p>transcrire l'audio en texte</p>
        <div
          className="file-input-wrapper"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file"
            className="file-input"
            onChange={handleFileChange}
          />
          <label htmlFor="file" className="custom-file-upload">
            Choose File or Drag and Drop Here
          </label>
          {audioFile ? (
            <p className="file-name">{audioFile.name}</p>
          ) : (
            <p className="file-name">No file selected</p>
          )}
        </div>
        <button onClick={handleUpload}>Upload</button>
        {error && (
          <div className="error-message">
            <p>Error:</p>
            <p>{error}</p>
          </div>
        )}
        {transcription && (
          <div className="transcription-result">
            <p>Transcription:</p>
            <p>{transcription}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
