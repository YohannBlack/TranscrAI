// TranslatePage.js

import React, { useState } from 'react';

const TranslatePage = () => {
  const [textToTranslate, setTextToTranslate] = useState("");
  const [fromLanguage, setFromLanguage] = useState("en"); // Langue par défaut en anglais
  const [toLanguage, setToLanguage] = useState("fr"); // Langue par défaut en français
  const [translationResult, setTranslationResult] = useState("");
  const [error, setError] = useState(null);

  const handleTranslate = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5000/api/v1/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: textToTranslate,
          from_language: fromLanguage,
          to_language: toLanguage,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setTranslationResult(data.translation);
      setError(null); // Reset error state on success
    } catch (error) {
      setError(error.message);
      console.error("Error translating text:", error);
    }
  };

  return (
    <div className="translate-page">
      <h1>Page de Traduction</h1>
      <div className="translate-form">
        <label htmlFor="textToTranslate">Texte à traduire:</label>
        <textarea
          id="textToTranslate"
          rows="4"
          value={textToTranslate}
          onChange={(e) => setTextToTranslate(e.target.value)}
        ></textarea>

        <label htmlFor="fromLanguage">De la langue:</label>
        <select
          id="fromLanguage"
          value={fromLanguage}
          onChange={(e) => setFromLanguage(e.target.value)}
        >
          <option value="en">Anglais</option>
          <option value="fr">Français</option>
          {/* Ajoutez d'autres options de langue selon vos besoins */}
        </select>

        <label htmlFor="toLanguage">Vers la langue:</label>
        <select
          id="toLanguage"
          value={toLanguage}
          onChange={(e) => setToLanguage(e.target.value)}
        >
          <option value="fr">Français</option>
          <option value="en">Anglais</option>
          {/* Ajoutez d'autres options de langue selon vos besoins */}
        </select>

        <button onClick={handleTranslate}>Traduire</button>
      </div>

      {translationResult && (
        <div className="translation-result">
          <h2>Résultat de la traduction:</h2>
          <p>{translationResult}</p>
        </div>
      )}

      {error && (
        <div className="error-message">
          <p>Erreur:</p>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default TranslatePage;
