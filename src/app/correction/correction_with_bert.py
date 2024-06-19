import torch
from transformers import BertTokenizer, BertForTokenClassification, AdamW
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

# Exemples d'entraînement
train_sentences = [
    "Je suis aller au marché.",
    "Il a manger une pomme.",
    "Nous avons danser toute la nuit.",
    "Vous êtes arriver en retard.",
    "Ils ont fini leur travail.",
    "Tu es parti trop tôt.",
    "Elle a pris une décision.",
    "Je veux aller à l'école.",
    "Il est temps de partir.",
    "Nous avons besoin d'aide.",
    "Je suis venu pour parler.",
    "Elle a acheter une robe.",
    "Nous avons jouer au football.",
    "Vous avez oublier votre sac.",
    "Ils ont courir rapidement.",
    "Tu as vu le film.",
    "Elle est rentrée tôt.",
    "Je suis allé chez moi.",
    "Il a fini ses devoirs.",
    "Nous avons décidé de partir."
]

train_labels = [
    [0, 0, 1, 0, 0],  # Je suis allé au marché.
    [0, 0, 1, 0, 0],  # Il a mangé une pomme.
    [0, 0, 1, 0, 0, 0],  # Nous avons dansé toute la nuit.
    [0, 0, 1, 0, 0],  # Vous êtes arrivés en retard.
    [0, 0, 0, 0, 0],  # Ils ont fini leur travail.
    [0, 0, 0, 0, 0],  # Tu es parti trop tôt.
    [0, 0, 0, 0, 0],  # Elle a pris une décision.
    [0, 0, 0, 0, 0, 0],  # Je veux aller à l'école.
    [0, 0, 0, 0, 0],  # Il est temps de partir.
    [0, 0, 0, 0, 0],  # Nous avons besoin d'aide.
    [0, 0, 0, 0, 0],  # Je suis venu pour parler.
    [0, 0, 1, 0, 0],  # Elle a acheté une robe.
    [0, 0, 1, 0, 0],  # Nous avons joué au football.
    [0, 0, 1, 0, 0],  # Vous avez oublié votre sac.
    [0, 0, 1, 0],  # Ils ont couru rapidement.
    [0, 0, 0, 0, 0],  # Tu as vu le film.
    [0, 0, 0, 0],  # Elle est rentrée tôt.
    [0, 0, 0, 0, 0],  # Je suis allé chez moi.
    [0, 0, 0, 0, 0],  # Il a fini ses devoirs.
    [0, 0, 0, 0, 0]   # Nous avons décidé de partir.
]

#0: pas de changement
#1: orthographe
#2: conjugaison
#3: accord
#4: autres

# Charger le tokenizer et le modèle
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=5)

# Prétraitement
encodings = tokenizer(train_sentences, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
labels = [label + [0] * (128 - len(label)) for label in train_labels]  # Padding des labels
labels = torch.tensor(labels)

dataset = CorrectionDataset(encodings, labels)

train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
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
