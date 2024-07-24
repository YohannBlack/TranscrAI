import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Translation from './translate';
import Transcription from './transcription';
import './styles/App.css'; // Assurez-vous que ce fichier existe et contient les styles globaux

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
