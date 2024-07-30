import argparse
from dotenv import load_dotenv
import os
import torch
from tqdm import tqdm
import json
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens.analysis.feature_statistics import get_W_U_W_dec_stats_df
import pandas as pd
from typing import Dict, List, Tuple


def load_model_and_sae(
    model_name: str, sae_release: str, sae_id: str, device: str = "cpu"
) -> Tuple[HookedTransformer, Dict[str, SAE], Dict[str, torch.Tensor]]:
    """
    Load the transformer model and sparse autoencoders.

    Args:
        model_name (str): Name of the transformer model to load.
        sae_release (str): Release name for the sparse autoencoder.
        sae_id (str): ID of the sparse autoencoder.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        Tuple containing the loaded model, dictionary of SAEs, and dictionary of SAE sparsities.
    """
    model = HookedTransformer.from_pretrained(model_name)
    sparse_autoencoders = {}
    sae_sparsities = {}

    for layer in range(model.cfg.n_layers):
        sae, _, sparsity = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device,
        )
        sparse_autoencoders[f"blocks.{layer}.hook_resid_pre"] = sae
        sae_sparsities[f"blocks.{layer}.hook_resid_pre"] = sparsity

    return model, sparse_autoencoders, sae_sparsities


def get_feature_statistics(
    model: HookedTransformer, sae: SAE, sparsity: torch.Tensor
) -> pd.DataFrame:
    """
    Calculate feature statistics for the given SAE.

    Args:
        model (HookedTransformer): The loaded transformer model.
        sae (SAE): The sparse autoencoder.
        sparsity (torch.Tensor): Sparsity values for the SAE features.

    Returns:
        pd.DataFrame: DataFrame containing feature statistics.
    """
    W_dec = sae.W_dec.detach().cpu()
    W_U_stats_df_dec, dec_projection_onto_W_U = get_W_U_W_dec_stats_df(
        W_dec, model, cosine_sim=False
    )
    W_U_stats_df_dec["sparsity"] = sparsity.cpu()
    return W_U_stats_df_dec, dec_projection_onto_W_U


def get_top_k_words(
    feature_activations: torch.Tensor, words: List[str], k: int = 10
) -> List[Tuple[str, float]]:
    """
    Get the top k activated words for a given feature.

    Args:
        feature_activations (torch.Tensor): Activation values for a feature.
        words (List[str]): List of words in the vocabulary.
        k (int): Number of top words to return.

    Returns:
        List[Tuple[str, float]]: List of tuples containing top words and their activation values.
    """
    if feature_activations.numel() == 0:
        return []

    k = min(k, feature_activations.numel())
    top_k_values, top_k_indices = torch.topk(feature_activations, k)
    top_k_words = [words[i] for i in top_k_indices.tolist()]
    top_k_activations = top_k_values.tolist()

    return list(zip(top_k_words, top_k_activations))


def process_features(
    dec_projection_onto_W_U: torch.Tensor, words: List[str]
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Process features to get top activated words for each feature.

    Args:
        dec_projection_onto_W_U (torch.Tensor): Decoder projection onto W_U.
        words (List[str]): List of words in the vocabulary.

    Returns:
        Dict[int, List[Tuple[str, float]]]: Dictionary mapping feature IDs to their top activated words.
    """
    top_activated_words = {}
    for i in tqdm(range(dec_projection_onto_W_U.shape[0]), desc="Processing features"):
        feature_activations = dec_projection_onto_W_U[i]
        top_activated_words[i] = get_top_k_words(feature_activations, words)
    return top_activated_words


def save_results(
    top_activated_words: Dict[int, List[Tuple[str, float]]], filename: str
):
    """
    Save the top activated words for each feature to a JSON file.

    Args:
        top_activated_words (Dict[int, List[Tuple[str, float]]]): Dictionary of top activated words per feature.
        filename (str): Name of the file to save the results.
    """
    json_output = {
        f"feature_{feature}": {"top_words": " ".join([word for word, _ in words])}
        for feature, words in top_activated_words.items()
    }

    with open(filename, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Results saved to '{filename}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from SAE")
    parser.add_argument(
        "--model_name", required=True, help="Name of the Transformer model"
    )
    parser.add_argument("--sae_release", required=True, help="The SAE release name")
    parser.add_argument("--sae_id", required=True, help="The ID of the SAE to load")
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="The SAE layer to extract features from",
    )
    parser.add_argument(
        "--output_file", required=True, help="Path to save the extracted features JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and SAE
    model, sparse_autoencoders, sae_sparsities = load_model_and_sae(
        args.model_name, args.sae_release, args.sae_id
    )

    # Focus on a specific layer (e.g., layer 8)
    layer = 8
    sae = sparse_autoencoders[f"blocks.{args.layer}.hook_resid_pre"]
    sparsity = sae_sparsities[f"blocks.{args.layer}.hook_resid_pre"]

    # Get feature statistics
    W_U_stats_df_dec, dec_projection_onto_W_U = get_feature_statistics(
        model, sae, sparsity
    )

    # Get vocabulary
    vocab = model.tokenizer.get_vocab()
    words = sorted(vocab.keys(), key=lambda x: vocab[x])

    # Process features
    top_activated_words = process_features(dec_projection_onto_W_U, words)

    # Save results
    save_results(top_activated_words, args.output_file)

    # Update DataFrame with top activated words
    W_U_stats_df_dec["top_activated_words"] = W_U_stats_df_dec.index.map(
        lambda i: " ".join([word for word, _ in top_activated_words[i]])
    )

    # Display the updated dataframe
    print(W_U_stats_df_dec)


if __name__ == "__main__":
    main()
