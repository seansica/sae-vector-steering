import torch
from dataset import PileDataset
from train_utils import get_dataloader, train_transformer, collect_mlp_activations
from models import OneLayerTransformer
import multiprocessing

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dataset and dataloader
    dataset = PileDataset(split="train", context_length=128)
    dataloader = get_dataloader(dataset, batch_size=32, num_workers=4)

    # Initialize and train transformer
    transformer = OneLayerTransformer(vocab_size=50257)  # GPT-2 vocab size
    train_transformer(transformer, dataloader, num_epochs=10, learning_rate=1e-4, device=device)

    # Collect MLP activations for autoencoder training
    activations = collect_mlp_activations(transformer, dataloader, num_samples=8_000_000, device=device)

    # The 'activations' tensor can now be used for autoencoder training

if __name__ == "__main__":
    # This is necessary for multiprocessing to work on macOS
    multiprocessing.set_start_method('spawn')
    main()