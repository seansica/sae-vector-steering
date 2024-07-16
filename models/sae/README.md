# Model Repository

This directory contains our trained models, including Sparse Autoencoders (SAEs) and fine-tuned language models. We use git-lfs to manage large model files.

## Directory Structure

```
models/
├── sae/
│   ├── <model_name>/
│   │   ├── model.pt
│   │   ├── config.json
│   │   └── model_card.md
│   └── ...
├── fine_tuned/
│   ├── <model_name>/
│   │   ├── model.pt
│   │   ├── config.json
│   │   └── model_card.md
│   └── ...
└── README.md
```

## Naming Convention

### For SAE models:
`<base_model>_<layer>_exp<expansion_factor>_<training_tokens>`

Example:
- `gpt2-small_mlp0_exp2_1M`: GPT-2 Small, MLP layer 0, expansion factor 2, trained on 1 million tokens

### For fine-tuned models:
`<base_model>_<domain>_<training_tokens>`

## Model Files

Each model folder should contain:
1. `model.pt`: The PyTorch saved model
2. `config.json`: Configuration used for training
3. `model_card.md`: Markdown file with model details

## Model Card Template

Use this template for `model_card.md`:

```markdown
# Model Card: [Model Name]

## Base Model
- Name: [e.g., GPT-2 Small]
- Source: [e.g., Hugging Face]

## SAE Configuration
- Expansion Factor: [Value]
- Target Layer: [e.g., MLP output of block 0]
- Input Dimension: [Value]

## Training
- Dataset: [Name]
- Tokens Processed: [Value]
- Batch Size: [Value]
- Context Size: [Value]
- Learning Rate: [Value]
- L1 Coefficient: [Value]

## Performance
- Final Loss: [Value]
- Feature Activation Rate: [Value]

## Intended Use
This model is intended for [purpose].

## Limitations
- [List any limitations]

## Additional Notes
[Any other relevant information]

## Date
[Date of training]

## Author
[Your Name/Organization]
```

## Contributing a New Model

To add a new model to this repository:

1. Ensure you have git-lfs installed and set up.
2. Create a new folder in the appropriate directory (`sae/` or `fine_tuned/`) using the naming convention.
3. Save your model as `model.pt` in this folder.
4. Create a `config.json` file with the training configuration.
5. Create a `model_card.md` file using the template above.
6. Commit your changes:
   ```
   git add models/<type>/<model_name>/
   git commit -m "Add new model: <model_name>"
   git push
   ```

Note: Large files (like `model.pt`) will be handled by git-lfs automatically.

## Using git-lfs

If you haven't set up git-lfs:

1. Install git-lfs: https://git-lfs.github.com/
2. In this repository, run: `git lfs install`
3. Track model files: `git lfs track "models/**/*.pt"`
4. Commit the `.gitattributes` file

Now you can commit and push model files as usual, and git-lfs will handle them appropriately.