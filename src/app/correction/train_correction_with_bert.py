import torch
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class CorrectionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_and_align_labels(sentences, labels):
    tokenized_inputs = tokenizer(sentences, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    label_ids = []

    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids.append([])
        for word_idx in word_ids:
            if word_idx is None:
                label_ids[-1].append(-100)
            elif word_idx != previous_word_idx:
                try:
                    label_ids[-1].append(label[word_idx])
                except IndexError:
                    label_ids[-1].append(-100)  # Étiquette par défaut pour les tokens non étiquetés
            else:
                label_ids[-1].append(-100)
            previous_word_idx = word_idx

    return tokenized_inputs, torch.tensor(label_ids)

# Exemples d'entraînement
train_sentences = [
    "I am go to the market.",
    "He has eat an apple.",
    "We have dance all night.",
    "You are arrive late.",
    "They have finished their work.",
    "You are left too early.",
    "She has took a decision.",
    "I want to go to school.",
    "It is time to leave.",
    "We need help.",
    "I am come to talk.",
    "She has buy a dress.",
    "We have play football.",
    "You have forget your bag.",
    "They have run quickly.",
    "You have saw the movie.",
    "She is come early.",
    "I have gone home.",
    "He has finished his homework.",
    "We have decided to leave."
]

train_labels = [
    [0, 0, 2, 0, 0, 0, 0],  # I am go to the market.
    [0, 0, 2, 0, 0, 0],  # He has eat an apple.
    [0, 0, 2, 0, 0, 0, 0],  # We have dance all night.
    [0, 0, 2, 0, 0],  # You are arrive late.
    [0, 0, 0, 0, 0, 0],  # They have finished their work.
    [0, 0, 2, 0, 0],  # You are left too early.
    [0, 0, 2, 0, 0],  # She has took a decision.
    [0, 0, 0, 0, 0, 0],  # I want to go to school.
    [0, 0, 0, 0, 0],  # It is time to leave.
    [0, 0, 0, 0],  # We need help.
    [0, 0, 2, 0, 0],  # I am come to talk.
    [0, 0, 2, 0, 0],  # She has buy a dress.
    [0, 0, 2, 0, 0],  # We have play football.
    [0, 0, 2, 0, 0, 0],  # You have forget your bag.
    [0, 0, 2, 0],  # They have run quickly.
    [0, 0, 2, 0, 0],  # You have saw the movie.
    [0, 0, 2, 0],  # She is come early.
    [0, 0, 0, 0],  # I have gone home.
    [0, 0, 0, 0, 0],  # He has finished his homework.
    [0, 0, 0, 0, 0]  # We have decided to leave.
]

#0: pas de changement
#1: orthographe
#2: conjugaison
#3: accord
#4: autres

# Charger le tokenizer fast et le modèle
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=5)

print("Tokenizer and model loaded.")

# Prétraitement
encodings, labels = tokenize_and_align_labels(train_sentences, train_labels)

dataset = CorrectionDataset(encodings, labels)

train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

print("DataLoader created.")

# Entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

def train_and_validate(model, train_dataloader, val_dataloader, optimizer, num_epochs, early_stopping_patience):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        print(f"Validation loss after epoch {epoch + 1}: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Sauvegarde du meilleur modèle
            model.save_pretrained("save/model")
            tokenizer.save_pretrained("save/tokenizer")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Visualiser les courbes de perte
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Appel de la fonction
train_and_validate(model, train_dataloader, val_dataloader, optimizer, num_epochs=20, early_stopping_patience=2)

print("Training completed and model saved.")
