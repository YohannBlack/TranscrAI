import torch
from transformers import BertTokenizer, BertForTokenClassification

# Charger le tokenizer et le modèle sauvegardés
tokenizer = BertTokenizer.from_pretrained('save/tokenizer')
model = BertForTokenClassification.from_pretrained('save/model')

# Exemples de test
test_sentences = [
    "Je suis aller au parc.",
    "Il a manger un gâteau.",
    "Nous avons chanter une chanson.",
    "Vous êtes arriver tôt.",
    "Ils ont dormi tôt.",
    "Tu es parti vite.",
    "Elle a pris son sac.",
    "Je veux aller à l'école.",
    "Il est temps de dormir.",
    "Nous avons besoin de toi."
]

# Prétraitement
def preprocess_sentences(sentences, tokenizer, max_length=128):
    encodings = tokenizer(sentences, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return encodings

encodings = preprocess_sentences(test_sentences, tokenizer)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Passer le modèle en mode évaluation
model.eval()

# Déplacer les tensors vers le GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Faire des prédictions
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Décoder les prédictions
def decode_predictions(predictions, input_ids, tokenizer):
    decoded_predictions = []
    for pred, input_id in zip(predictions, input_ids):
        tokens = tokenizer.convert_ids_to_tokens(input_id)
        corrected_sentence = []
        for token, label in zip(tokens, pred):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            corrected_sentence.append(token)
        decoded_predictions.append(" ".join(corrected_sentence))
    return decoded_predictions

corrected_sentences = decode_predictions(predictions, input_ids, tokenizer)

# Afficher les résultats
for original, corrected in zip(test_sentences, corrected_sentences):
    print(f"Original: {original}")
    print(f"Corrected: {corrected}")
    print()
