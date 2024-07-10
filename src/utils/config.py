config = {
    "output_file": "tokenized_pile.h5",
    "chunk_size": 1000000,
    "max_tokens": 1_000_000,  # Process 1 million tokens
    # "max_tokens": 1_000_000_000,  # Process 1 billion tokens
    "max_samples": None,  # Set to an integer if you want to limit by number of samples
    "context_length": 1024,  # Adjust this based on your model's context length
}
