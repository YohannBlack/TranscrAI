import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

print("Loading tokenizer and model...")
# Charger le tokenizer et le modèle pré-entrainé pour la correction grammaticale
tokenizer = T5Tokenizer.from_pretrained("save/t5_tokenizer")
model = T5ForConditionalGeneration.from_pretrained("save/t5_model")
print("Tokenizer and model loaded.")

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

print("Tokenizing test sentences...")
# Prétraiter les phrases de test
inputs = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True)
print("Test sentences tokenized.")

print("Generating corrections...")
# Générer les corrections
outputs = model.generate(inputs["input_ids"], max_length=128)
print("Corrections generated.")

# Décoder les corrections
corrected_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Afficher les corrections
for i, sentence in enumerate(test_sentences):
    print(f"Original: {sentence}")
    print(f"Corrected: {corrected_sentences[i]}")
    print()
