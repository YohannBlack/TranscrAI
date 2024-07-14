import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class CorrectionDataset(Dataset):
    def __init__(self, input_texts, target_texts, tokenizer, max_len):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        input_encoding = self.tokenizer(
            input_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding["input_ids"].flatten(),
            'attention_mask': input_encoding["attention_mask"].flatten(),
            'labels': labels.flatten()
        }

# Exemples d'entraînement
# Phrases avec erreurs
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
    "We have decided to leave.",
    "He has eat an apple.",
    "She have go to the store.",
    "We is going to the movies.",
    "He are playing in the park.",
    "I has read the book.",
    "You is running very fast.",
    "She has forgot her keys.",
    "They have decided to stay home.",
    "I am not understanding the question.",
    "You has made a mistake.",
    "She has make a decision.",
    "They is watching TV.",
    "He has broke the vase.",
    "I has seen the movie.",
    "You is being silly.",
    "We have drink all the juice.",
    "She is go to school.",
    "They has finished the project.",
    "He have take my pen.",
    "I am write a letter.",
    "You has buy a car.",
    "She have find her phone.",
    "We is reading a book.",
    "They have sing a song.",
    "He are work hard.",
    "I has eat my lunch.",
    "You have break the rules.",
    "She is going to the park.",
    "They has decided to go.",
    "We have make a cake."
]

# Phrases corrigées
corrected_sentences = [
    "I am going to the market.",
    "He has eaten an apple.",
    "We have danced all night.",
    "You have arrived late.",
    "They have finished their work.",
    "You left too early.",
    "She has made a decision.",
    "I want to go to school.",
    "It is time to leave.",
    "We need help.",
    "I have come to talk.",
    "She has bought a dress.",
    "We have played football.",
    "You have forgotten your bag.",
    "They have run quickly.",
    "You have seen the movie.",
    "She has come early.",
    "I have gone home.",
    "He has finished his homework.",
    "We have decided to leave.",
    "He has eaten an apple.",
    "She has gone to the store.",
    "We are going to the movies.",
    "He is playing in the park.",
    "I have read the book.",
    "You are running very fast.",
    "She has forgotten her keys.",
    "They have decided to stay home.",
    "I do not understand the question.",
    "You have made a mistake.",
    "She has made a decision.",
    "They are watching TV.",
    "He has broken the vase.",
    "I have seen the movie.",
    "You are being silly.",
    "We have drunk all the juice.",
    "She is going to school.",
    "They have finished the project.",
    "He has taken my pen.",
    "I am writing a letter.",
    "You have bought a car.",
    "She has found her phone.",
    "We are reading a book.",
    "They have sung a song.",
    "He is working hard.",
    "I have eaten my lunch.",
    "You have broken the rules.",
    "She is going to the park.",
    "They have decided to go.",
    "We have made a cake."
]


# Charger le tokenizer et le modèle T5
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Prétraitement
max_len = 50
train_dataset = CorrectionDataset(train_sentences, corrected_sentences, tokenizer, max_len)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

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
            model.save_pretrained("save/t5_model")
            tokenizer.save_pretrained("save/t5_tokenizer")
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
