# Model Card: gpt2-small_mlp0_exp2_1M

## Base Model
- Name: GPT-2 Small
- Source: Hugging Face

## SAE Configuration
- Expansion Factor: 2
- Target Layer: MLP output of block 0
- Input Dimension: 768

## Training
- Dataset: Pile Uncopyrighted
- Tokens Processed: 1 million
- Batch Size: 1024 tokens
- Context Size: 128 tokens
- Learning Rate: 1e-4
- L1 Coefficient: 1.0

## Performance
- Final Loss: [To be filled after training]
- Feature Activation Rate: [To be filled after training]

## Intended Use
This model is intended as a proof of concept for training a Sparse Autoencoder (SAE) on the activations of GPT-2 Small's first MLP layer. It can be used for preliminary feature analysis and as a starting point for more comprehensive SAE training.

## Limitations
- Trained on a small subset of data (1 million tokens), which may not be representative of the full dataset.
- Uses a small expansion factor (2x) compared to more comprehensive SAEs, potentially limiting its ability to capture a wide range of features.
- Trained for a short duration, which may result in incomplete feature learning.
- May not capture all relevant features due to limited training data and time.

## Additional Notes
This SAE was trained as a quick proof of concept on an MPS device (Apple Silicon). The primary goal was to verify the training process, saving, and restoration of the SAE. For more robust feature analysis, consider training for longer, with more data, and a larger expansion factor.

## Date
2024-07-16

## Author
Sean Sica