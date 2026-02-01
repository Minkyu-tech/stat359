import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Device selection: CUDA > MPS > CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using - {device}")

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    """ Transform the dataset as tensor """
    def __init__(self, pairs):
        self.data = torch.tensor(pairs, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    """
    Two nn.Embedding layers
    - input: center words
    - output: context words

    forword: return dot product
    """
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.input_embed = nn.Embedding(vocab_size, embedding_dim)  # Input: Center words
        self.output_embed = nn.Embedding(vocab_size, embedding_dim)  # Output: Context words

    def forward(self, center_words, context_words):
        center_vectors = self.input_embed(center_words)
        context_vectors = self.output_embed(context_words)
        logits = (center_vectors * context_vectors).sum(dim=1)  # Dot product
        return logits
    
    def get_embeddings(self):
        return self.input_embed.weight.data.cpu().numpy()

# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# dict_keys(['sent_list', 'counter', 'word2idx', 'idx2word', 'skipgram_df'])
word2idx = data['word2idx']
idx2word = data['idx2word']
skipgram_pairs = data['skipgram_df'].values

vocab_size = len(word2idx)
print(f"Vocab Size: {vocab_size}")  # 18,497
print(f"Number of pairs: {len(skipgram_pairs)}")  # 18,959,430

# Precompute negative sampling distribution below
## Step 1: Obtain word frequency counts
counter = data['counter']

## Step 2: Create a tensor of word counts aligned with vocab indices
counts_tensor = torch.zeros(vocab_size)
for word, idx in word2idx.items():
    cnt = counter.get(word, 0) 
    counts_tensor[idx] = cnt
print(f"Counts tensor: {counts_tensor.shape}")  # torch.Size([18497])

## Step 3: Apply 3/4 power smoothing and normalize
pow_counts = counts_tensor.pow(0.75)  # P(w) = count(w)^0.75
total_pow = pow_counts.sum()
unigram_dist = pow_counts / total_pow
print(f"Dist. Sum: {unigram_dist.sum().item():.2f}") # 1.00

# Step 4: Use torch.multinomial() to sample negative context words
def make_targets(center, context, vocab_size):
    batch_size = center.size(0)
    device = center.device

    num_neg = batch_size * NEGATIVE_SAMPLES

    neg_contexts = torch.multinomial(unigram_dist, num_neg, replacement=True).to(device)
    neg_centers = center.repeat_interleave(NEGATIVE_SAMPLES)

    final_center = torch.cat([center, neg_centers])
    final_context = torch.cat([context, neg_contexts])

    pos_pairs = torch.ones(batch_size, device=device)  # Label = 1
    neg_pairs = torch.zeros(num_neg, device=device)  # Label = 0
    final_targets = torch.cat([pos_pairs, neg_pairs])
    
    return final_center, final_context, final_targets

# Model, Loss, Optimizer
## Step 5: Use BCEWithLogitsLoss to compute the combined loss
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_pairs)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Training loop
print(">>> Training Starts <<<")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for center, context in progress_bar:
        center = center.to(device)
        context = context.to(device)

        final_center, final_context, final_targets = make_targets(center, context, vocab_size)
        
        logits = model.forward(final_center, final_context)
        loss = criterion(logits, final_targets.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

print("Training Compeleted.")

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({
        'embeddings': embeddings,
        'word2idx': data['word2idx'],
        'idx2word': data['idx2word']
        }, f)
print("Embeddings saved to word2vec_embeddings.pkl")
