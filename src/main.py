import argparse
import os
from dotenv import load_dotenv
import subprocess

import torch


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def load_config():
    # Load .env file
    load_dotenv()

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the SAE feature extraction and steering pipeline")
    parser.add_argument("--device", help="Manually specify the device from [cuda, mps, cpu]", default=get_device())
    parser.add_argument("--model_name", help="Name of the Transformer model")
    parser.add_argument("--sae_release", help="The SAE release name")
    parser.add_argument("--sae_id", help="The ID of the SAE to load")
    parser.add_argument("--layer", type=int, help="The SAE layer to extract features from")
    parser.add_argument("--extracted_features_filename", help="Path to the JSON file for extracted features",)
    parser.add_argument("--feature_to_steer", type=int, help="Feature ID to use for steering")
    parser.add_argument("--steering_coeff", type=float, help="Coefficient for steering")
    parser.add_argument("--steering_temp", type=float, help="Temperature for steering")
    parser.add_argument("--steering_prompt", help="Prompt for steering")

    args = parser.parse_args()

    # Prioritize CLI arguments over environment variables
    config = {
        "DEVICE": args.device or "cpu",
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"), # not supported as a CLI flag for security purposes
        "MODEL_NAME": args.model_name or os.getenv("MODEL_NAME"),
        "SAE_RELEASE": args.sae_release or os.getenv("SAE_RELEASE"),
        "SAE_ID": args.sae_id or os.getenv("SAE_ID"),
        "LAYER": args.layer or int(os.getenv("LAYER")),
        "EXTRACTED_FEATURES_FILE": args.extracted_features_filename or os.getenv("EXTRACTED_FEATURES_FILE"),
        "INTERPRETED_FEATURES_FILE": args.interpreted_features_filename or os.getenv("INTERPRETED_FEATURES_FILE"),
        "FEATURE_TO_STEER": args.feature_to_steer or int(os.getenv("FEATURE_TO_STEER")),
        "STEERING_COEFF": args.steering_coeff or float(os.getenv("STEERING_COEFF")),
        "STEERING_TEMP": args.steering_temp or float(os.getenv("STEERING_TEMP")),
        "STEERING_PROMPT": args.steering_prompt or os.getenv("STEERING_PROMPT"),
    }

    return config


def run_extract_features(config):
    cmd = [
        "python", "0_extract_features.py", 
        "--model_name", config["MODEL_NAME"],
        "--sae_release", config["SAE_RELEASE"],
        "--sae_id", config["SAE_ID"],
        "--layer", str(config["LAYER"]),
        "--output_file", config["EXTRACTED_FEATURES_FILENAME"],
    ]
    subprocess.run(cmd, check=True)


def run_interpret_features(config):
    cmd = [
        "python", "1_interpret_features.py",
        "--input_file", config["EXTRACTED_FEATURES_FILE"],
        "--output_file", config["INTERPRETED_FEATURES_FILE"],
        "--openai_api_key", config["OPENAI_API_KEY"],
        "interpreted_features.json",
    ]
    subprocess.run(cmd, check=True)


def run_apply_vector_steering(config):
    cmd = [
        "python", "2_apply_vector_steering.py",
        "--model_name", config["MODEL_NAME"],
        "--sae_release", config["SAE_RELEASE"],
        "--sae_id", config["SAE_ID"],
        "--layer", str(config["LAYER"]),
        "--feature_id", str(config["FEATURE_TO_STEER"]),
        "--coeff", str(config["STEERING_COEFF"]),
        "--temperature", str(config["STEERING_TEMP"]),
        "--prompt", config["STEERING_PROMPT"],
    ]
    subprocess.run(cmd, check=True)


def main():
    config = load_config()

    print("Using device:", config["DEVICE"])

    print("Running feature extraction...")
    run_extract_features(config)

    print("Running feature interpretation...")
    run_interpret_features(config)

    print("Applying vector steering...")
    run_apply_vector_steering(config)


if __name__ == "__main__":
    main()
