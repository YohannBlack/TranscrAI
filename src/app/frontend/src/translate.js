import React, { useState } from "react";
import "./styles/translate.css"; // Importez le fichier CSS pour les styles

const TranslatePage = () => {
  const [textToTranslate, setTextToTranslate] = useState("");
  const [fromLanguage, setFromLanguage] = useState("en");
  const [toLanguage, setToLanguage] = useState("fr");
  const [translationResult, setTranslationResult] = useState("");
  const [error, setError] = useState(null);

  const handleTranslate = async () => {
    const formData = new FormData();
    formData.append("text", textToTranslate);
    formData.append("from_language", fromLanguage);
    formData.append("to_language", toLanguage);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/v1/translate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setTranslationResult(data[0].translation);
      setError(null); // Reset error state on success
    } catch (error) {
      setError(error.message);
      console.error("Error translating text:", error);
    }
  };

  const handleCopyResult = () => {
    navigator.clipboard
      .writeText(translationResult)
      .then(() => {
        // No need to show success message
      })
      .catch((err) =>
        console.error("Erreur lors de la copie du texte : ", err)
      );
  };

  return (
    <div className="translate-page">
      <h1>Traducteur</h1>
      <div className="translate-form">
        <div className="form-group">
          <label htmlFor="textToTranslate">Texte à traduire:</label>
          <textarea
            id="textToTranslate"
            rows="4"
            value={textToTranslate}
            onChange={(e) => setTextToTranslate(e.target.value)}></textarea>
        </div>

        <div className="form-group">
          <label htmlFor="fromLanguage">De la langue:</label>
          <select
            id="fromLanguage"
            value={fromLanguage}
            onChange={(e) => setFromLanguage(e.target.value)}>
            <option value="en">English</option>
            <option value="fr">French</option>
            {/* Ajoutez d'autres options de langue selon vos besoins */}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="toLanguage">Vers la langue:</label>
          <select
            id="toLanguage"
            value={toLanguage}
            onChange={(e) => setToLanguage(e.target.value)}>
            <option value="fr">French</option>
            <option value="en">English</option>
            {/* Ajoutez d'autres options de langue selon vos besoins */}
          </select>
        </div>

        <button className="translate-button" onClick={handleTranslate}>
          Traduire
        </button>
      </div>

      {translationResult && (
        <div className="translation-result">
          <h2>Résultat de la traduction:</h2>
          <div>{translationResult}</div>
          <button className="copy-button" onClick={handleCopyResult}>
            Copier
          </button>
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
