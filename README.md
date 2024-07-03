# Polysemanticity & Bias Reduction in Language Models

## Course Information
- **Class**: DATASCI266
- **Section**: Summer 2024 Section 3
- **Group Members**: Sean Sica and Rini Gupta

## Project Overview

This project aims to explore the relationship between polysemanticity and gender bias in language models, specifically focusing on transformer-based sparse autoencoder models.

We propose a tehnique to reduce a model's gender bias by systematically reducing polysemanticity and hand-tuning isolated, monosemantic features correlated with gender bias terms.

This research is inspired by Anthropic's [Towards Monosematicity](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder) paper.

## Repository Structure

- `src/`: Source code for the project
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `scratch/`: Throwaway notebooks and scripts used for initial proofs of concept
- `scripts/`: Utility scripts for running experiments
- `configs/`: Configuration files

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Review the configuration files in `configs/`

## Main Components

1. **Transformer Model**: A custom implementation of a transformer-based language model.
2. **Sparse Autoencoder**: Used for feature extraction and polysemanticity reduction.
3. **Gender Bias Evaluation**: Framework for assessing and quantifying gender bias in the model.
4. **Polysemanticity Reduction Techniques**: Methods to encourage monosemantic features in the model.

## Current Status

This project is currently in development. Please check the issues and project boards for the most up-to-date status and ongoing tasks.

## Contributing

This is an academic project for DATASCI266. Contributions are limited to the assigned group members and course instructors.

## License

TBD