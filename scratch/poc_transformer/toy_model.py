import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import math

# Constants
CONTEXT_LENGTH = 128
BATCH_SIZE = 8192
VOCAB_SIZE = 50257  # GPT-2 tokenizer vocabulary size
RESIDUAL_STREAM_DIM = 128
MLP_DIM = 512
NUM_HEADS = 1
AUTOENCODER_HIDDEN_DIM = 1024  # Adjust as needed
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10  # Adjust based on your computational resources
RESAMPLING_STEPS = [25000, 50000, 75000, 100000]

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

tokenizer = tiktoken.get_encoding("gpt2")

class OneLayerTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, mlp_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x

    def get_mlp_activation(self, x):
        x = self.embedding(x) + self.pos_encoding(x)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        return self.mlp[0](x)  # Return activation after first linear layer and ReLU

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

class PileDataset(Dataset):
    def __init__(self, dataset_iterator, context_length):
        self.dataset_iterator = dataset_iterator
        self.context_length = context_length
        self.current_tokens = []

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.current_tokens) < self.context_length:
            try:
                item = next(self.dataset_iterator)
                text = item['text'] + "<|endoftext|>"
                self.current_tokens.extend(tokenizer.encode(text, allowed_special={"<|endoftext|>"}))
            except StopIteration:
                if len(self.current_tokens) < self.context_length:
                    raise StopIteration

        tokens = self.current_tokens[:self.context_length]
        self.current_tokens = self.current_tokens[self.context_length:]

        return torch.tensor(tokens, dtype=torch.long)

def train_transformer(model, dataloader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.view(-1, VOCAB_SIZE), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

def collect_mlp_activations(model, dataloader, num_samples):
    model.eval()
    activations = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting MLP activations"):
            batch = batch.to(device)
            batch_activations = model.get_mlp_activation(batch)
            activations.append(batch_activations.cpu())
            if len(activations) * BATCH_SIZE >= num_samples:
                break
    return torch.cat(activations, dim=0)[:num_samples]

def resample_dead_neurons(autoencoder, activations):
    with torch.no_grad():
        hidden = autoencoder.relu(autoencoder.encoder(activations))
        dead_neurons = (hidden.sum(dim=0) == 0).nonzero().squeeze()
        
        if dead_neurons.numel() > 0:
            print(f"Resampling {dead_neurons.numel()} dead neurons")
            losses = ((autoencoder(activations)[0] - activations) ** 2).mean(dim=1)
            probs = losses ** 2
            probs /= probs.sum()
            
            for neuron in dead_neurons:
                sample_idx = torch.multinomial(probs, 1)
                sample = activations[sample_idx].squeeze()
                
                autoencoder.encoder.weight[neuron] = sample / sample.norm() * 0.2
                autoencoder.encoder.bias[neuron] = 0
                autoencoder.decoder.weight[:, neuron] = sample / sample.norm()

def train_autoencoder(autoencoder, activations, num_steps):
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for step in tqdm(range(num_steps), desc="Training autoencoder"):
        idx = torch.randperm(activations.size(0))[:BATCH_SIZE]
        batch = activations[idx].to(device)

        optimizer.zero_grad()
        decoded, encoded = autoencoder(batch)
        loss = criterion(decoded, batch) + 1e-3 * encoded.abs().mean()  # L1 regularization
        loss.backward()
        optimizer.step()

        if step + 1 in RESAMPLING_STEPS:
            resample_dead_neurons(autoencoder, activations)

        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")

# Main execution
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
pile_dataset = PileDataset(iter(dataset), CONTEXT_LENGTH)
dataloader = DataLoader(pile_dataset, batch_size=BATCH_SIZE, num_workers=4)

transformer = OneLayerTransformer(VOCAB_SIZE, RESIDUAL_STREAM_DIM, NUM_HEADS, MLP_DIM).to(device)
train_transformer(transformer, dataloader, NUM_EPOCHS)

activations = collect_mlp_activations(transformer, dataloader, 8_000_000)  # Collect 8 million activations

autoencoder = Autoencoder(MLP_DIM, AUTOENCODER_HIDDEN_DIM).to(device)
train_autoencoder(autoencoder, activations, 1_000_000)  # Train for 1 million steps

torch.save(transformer.state_dict(), "transformer_model.pth")
torch.save(autoencoder.state_dict(), "autoencoder_model.pth")
print("Models saved successfully.")
