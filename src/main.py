import os
import torch
from datasets import load_dataset
import tiktoken
from torch.utils.data import DataLoader

from data.pile_dataset import PileDataset
from models.transformer import Transformer
from models.autoencoder import SparseAutoencoder
from training.train_transformer import train_transformer
from training.train_autoencoder import train_autoencoder
from utils.config import load_config
from utils.visualizations import plot_loss, plot_features

def main():

    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load configuration
    config = load_config("config.json")

    # Load dataset
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("r50k_base")

    # Create PileDataset instance
    pile_dataset = PileDataset(iter(dataset), config["context_length"], tokenizer)

    # Split dataset
    train_dataset, val_dataset, test_dataset = pile_dataset.split(0.8, 0.1, 0.1)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=4)

    # Initialize models
    transformer = Transformer(config).to(device)
    autoencoder = SparseAutoencoder(config).to(device)

    # Train transformer
    transformer, transformer_losses = train_transformer(
        transformer, train_loader, val_loader, config, device
    )

    # Train autoencoder
    autoencoder, autoencoder_losses = train_autoencoder(
        autoencoder, transformer, train_loader, config, device
    )

    # Plot losses
    plot_loss(transformer_losses, "transformer_loss.png")
    plot_loss(autoencoder_losses, "autoencoder_loss.png")

    # Plot features
    plot_features(autoencoder, "autoencoder_features.png")

if __name__ == "__main__":
    main()
