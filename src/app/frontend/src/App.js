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

        output += new TextDecoder("utf-8").decode(value);
        setTranscription((prevTranscription) => prevTranscription + output);
      }
      setError(null); // Réinitialisez l'état d'erreur en cas de succès
    } catch (error) {
      setError(error.message);
      console.error("Error uploading file:", error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <p>Upload an audio file to transcribe:</p>
        <input type="file" onChange={handleFileChange} />
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
