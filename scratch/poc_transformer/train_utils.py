import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_transformer(model, dataloader, num_epochs, learning_rate, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def collect_mlp_activations(model, dataloader, num_samples, device):
    model.eval()
    activations = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting MLP activations"):
            batch = batch.to(device)
            batch_activations = model.get_mlp_activation(batch)
            activations.append(batch_activations.cpu())
            if sum(a.size(0) for a in activations) >= num_samples:
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

def train_autoencoder(autoencoder, activations, num_steps, batch_size, learning_rate, device, resampling_steps):
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for step in tqdm(range(num_steps), desc="Training autoencoder"):
        idx = torch.randperm(activations.size(0))[:batch_size]
        batch = activations[idx].to(device)

        optimizer.zero_grad()
        decoded, encoded = autoencoder(batch)
        loss = criterion(decoded, batch) + 1e-3 * encoded.abs().mean()  # L1 regularization
        loss.backward()
        optimizer.step()

        if step + 1 in resampling_steps:
            resample_dead_neurons(autoencoder, activations)

        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")

def collate_batch(batch):
    return torch.stack(batch)

def get_dataloader(dataset, batch_size, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )