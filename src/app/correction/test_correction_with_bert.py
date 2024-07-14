import torch
from transformers import BertTokenizer, BertForTokenClassification

# Charger le tokenizer et le modèle finement entraîné
tokenizer = BertTokenizer.from_pretrained("save/tokenizer")
model = BertForTokenClassification.from_pretrained("save/model")

# Les phrases à tester
test_sentences = [
    "He has eat an apple.",
    "She have go to the store.",
    "We is going to the movies.",
    "He are playing in the park.",
    "I has read the book.",
    "You is running very fast.",
    "She has forgot her keys.",
    "They have decided to stay home.",
    "I am not understanding the question.",
    "You has made a mistake."
]

# Prétraiter les phrases de test
encodings = tokenizer(test_sentences, truncation=True, padding=True, return_tensors="pt")

# Passer en mode évaluation
model.eval()

# Effectuer des prédictions
with torch.no_grad():
    outputs = model(input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"])
    predictions = torch.argmax(outputs.logits, dim=2)

# Décoder les prédictions
for i, sentence in enumerate(test_sentences):
    predicted_labels = predictions[i].tolist()
    tokens = tokenizer.tokenize(sentence)
    corrected_sentence = []
    for token, label in zip(tokens, predicted_labels):
        if label == 0:  # Aucun changement
            corrected_sentence.append(token)
        elif label == 1:  # Correction d'orthographe
            corrected_sentence.append(token + " (orth)")
        elif label == 2:  # Conjugaison incorrecte
            corrected_sentence.append(token + " (conj)")
        elif label == 3:  # Accord incorrect
            corrected_sentence.append(token + " (acc)")
        elif label == 4:  # Autres erreurs
            corrected_sentence.append(token + " (autr)")

    corrected_sentence = tokenizer.convert_tokens_to_string(corrected_sentence)
    print(f"Original: {sentence}")
    print(f"Corrected: {corrected_sentence}")
    print()

