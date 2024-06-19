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

const App = () => {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route
            path="/"
            element={
              <div className="home-page">
                <main className="home-main">
                  <div className="instruction-text">
                    <h2>Que voulez-vous faire ?</h2>
                  </div>
                  <nav className="nav-container">
                    <ul className="nav-list">
                      <li className="nav-item">
                        <Link to="/transcription">
                          <div className="nav-block">
                            <img src={require('./image/transcrire.png')} alt="Transcription" className="nav-image" />
                          </div>
                        </Link>
                      </li>
                      <li className="nav-item">
                        <Link to="/translation">
                          <div className="nav-block">
                            <img src={require('./image/traduire.png')} alt="Traduction" className="nav-image" />
                          </div>
                        </Link>
                      </li>
                    </ul>
                  </nav>
                </main>
              </div>
            }
          />
          <Route path="/translation" element={<Translation />} />
          <Route path="/transcription" element={<Transcription />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
