#!/usr/bin/env python
# coding: utf-8
"""
Script for training an LSTM-based sentiment classifier on the financial_phrasebank dataset.
"""

# ========== Imports ==========
import os
import re
import numpy as np
import pandas as pd
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from gensim.models import KeyedVectors


print("\n========== Loading Dataset ==========")
# ========== Load Dataset ==========
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")


# ========== FastText Loading ==========
print("\n========== Loading FastText Vectors ==========")
FASTTEXT_PATH = os.path.join("models", "fasttext-wiki-news-subwords-300.model")
if not os.path.exists(FASTTEXT_PATH):
    raise FileNotFoundError(
        f"FastText model not found at: {FASTTEXT_PATH}\n"
        "Run this script from the assignment root directory so 'models/' is found."
    )

ft = KeyedVectors.load(FASTTEXT_PATH, mmap="r")
EMB_DIM = ft.vector_size
print(f"FastText loaded. vector_size={EMB_DIM}")


# ========== Tokenization + LSTM Input Encoding ==========
SEQ_LEN = 32  # handout requirement (pad/truncate)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?")

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def simple_tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text)]

def sentence_to_seq_fasttext(sentence: str, ft: KeyedVectors, seq_len: int = 32, dim: int = 300):
    tokens = simple_tokenize(sentence)

    seq = np.zeros((seq_len, dim), dtype=np.float32)
    if len(tokens) == 0:
        return seq

    tokens = tokens[:seq_len]
    for i, tok in enumerate(tokens):
        try:
            seq[i] = ft.get_vector(tok).astype(np.float32)
        except KeyError:
            # if the keyedvectors doesn't support OOV inference for that token
            seq[i] = 0.0
    return seq

def encode_sentences_lstm(sentences, ft: KeyedVectors, seq_len: int = 32, dim: int = 300):
    X = []
    for s in tqdm(list(sentences), desc=f"Encoding (FastText -> ({seq_len},{dim}))"):
        X.append(sentence_to_seq_fasttext(s, ft, seq_len=seq_len, dim=dim))
    return np.stack(X)  # (N, seq_len, dim)


print("\n========== Encoding Sentences for LSTM ==========")
X = encode_sentences_lstm(data['sentence'], ft, seq_len=SEQ_LEN, dim=EMB_DIM)
y = data['label'].values
print(f"X shape: {X.shape} (N, {SEQ_LEN}, {EMB_DIM}), y shape: {y.shape}")


# ========== Train/Test Split ==========
print("\n========== Splitting Data ==========")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")


# ========== PyTorch Dataset ==========
class LSTMDataset(Dataset):
    def __init__(self, X, labels):
        self.X = torch.tensor(X, dtype=torch.float32)       # (N, 32, 300)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'x': self.X[idx],          # (32, 300)
            'labels': self.labels[idx]
        }

train_dataset = LSTMDataset(X_train, y_train)
val_dataset = LSTMDataset(X_val, y_val)
test_dataset = LSTMDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("DataLoaders created.")


# ========== Model Definition ==========
print("\n========== Defining LSTM Classifier ==========")
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        return self.fc(out)

num_classes = len(np.unique(y))
model = LSTMClassifier(input_dim=EMB_DIM, hidden_dim=128, num_layers=1, num_classes=num_classes, dropout=0.3)


# ========== Training Setup ==========
print("\n========== Setting Up Training ==========")
device = get_device()
print(f"Using device: {device}")
os.makedirs("outputs", exist_ok=True)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

counts = [684, 2879, 1363] 
class_weights = 1. / torch.tensor(counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Training setup complete.")

# ========== Training Loop ==========
print("\n========== Starting Training Loop ==========")
num_epochs = 30 
best_val_f1 = 0.0
train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        x = batch['x'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(x)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.detach().cpu().numpy())
        all_train_labels.extend(labels.detach().cpu().numpy())

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
    train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).mean()
    train_loss_history.append(epoch_train_loss)
    train_f1_history.append(train_f1)
    train_acc_history.append(train_acc)
    print(f"Train Loss: {epoch_train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    all_val_preds, all_val_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
            x = batch['x'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(x)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            all_val_preds.extend(preds.detach().cpu().numpy())
            all_val_labels.extend(labels.detach().cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_acc = (np.array(all_val_preds) == np.array(all_val_labels)).mean()
    val_loss_history.append(epoch_val_loss)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)
    print(f"Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'outputs/best_lstm_model.pth')
        print(f'>>> Saved new best model (Val F1: {best_val_f1:.4f})')


# ========== Plot Learning Curves ==========
print("\n========== Plotting Learning Curves ==========")
plt.figure(figsize=(12, 15))
plt.subplot(3, 1, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.title('F1 Macro Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/lstm_f1_learning_curves.png')
plt.show()
print("Learning curves saved as 'outputs/lstm_f1_learning_curves.png'.")

# Save accuracy plot separately
plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/lstm_accuracy_learning_curve.png')
plt.show()
print("Accuracy curve saved as 'outputs/lstm_accuracy_learning_curve.png'.")


# ========== Test Evaluation ==========
print("\n========== Evaluating on Test Set ==========")
model.load_state_dict(torch.load('outputs/best_lstm_model.pth', map_location=device))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        x = batch['x'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

print('\n' + '='*50)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1_macro:.4f}")
print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
print('='*50 + '\n')

class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('outputs/lstm_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'outputs/lstm_confusion_matrix.png'.")

print("\nPer-class F1 Scores:")
for i, name in enumerate(class_names):
    class_f1 = f1_score(all_labels, all_preds, labels=[i], average='macro')

print("\n========== Script Complete ==========")
# End of script