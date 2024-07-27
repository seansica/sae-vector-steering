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

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Login to HuggingFace: `huggingface-cli login --token $YOUR_TOKEN`
4. ...

### 0. Extracting Features

The first step is extracting the features and finding the topK activations to contextualize the features.

```bash
python3 src/0_extract_features.py \
  --model_name gpt2-small \
  --sae_release gpt2-small-res-jb \
  --sae_id blocks.0.hook_resid_pre \
  --layer 8 \
  --output_file top_activations_per_feature.json
```

or if you want to load from the dotenv values:

```bash
source .env;

python3 src/0_extract_features.py --model_name $MODEL_NAME --sae_release $SAE_RELEASE --sae_id $SAE_ID --layer $LAYER --output_file $TOP_ACTIVATIONS_FILE
```

Expected output:
```
Loaded pretrained model gpt2-small into HookedTransformer
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
blocks.0.hook_resid_pre/cfg.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1.27k/1.27k [00:00<00:00, 2.25MB/s]
sae_weights.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 151M/151M [00:01<00:00, 78.5MB/s]
sparsity.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 98.4k/98.4k [00:00<00:00, 67.7MB/s]
Processing features: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24576/24576 [00:20<00:00, 1184.03it/s]
Results saved to 'top_activations_per_feature.json'
       feature          mean       std  skewness   kurtosis   sparsity                                top_activated_words
0            0  1.707836e-10  0.189715  0.764197   9.097904  -3.760490  ĠBrigade Ġbrigade ĠBrig Ġbrig Ġbattalion ĠBatt...
1            1 -8.804840e-09  0.166483  0.083962   3.300045 -10.000000  existent ASC arius ouses ucking ensical aryn a...
2            2 -7.590380e-11  0.203762  0.698958   6.764231  -3.327929  ĠMRI MRI Ġultrasound Ġautopsy Ġmicroscope Ġmic...
3            3  1.973499e-09  0.189168  0.627483   8.012758  -3.775403  Ġcontingent Ġconting Ġcontingency Ġconditioned...
4            4 -1.525666e-08  0.234610  1.676332  10.387130  -3.804149                      40 50 60 80 30 70 20 41 45 39
...        ...           ...       ...       ...        ...        ...                                                ...
24571    24571  1.785162e-08  0.164352  0.128330   3.678063  -4.767260  Land ĠSub iors anka Else ĠSkies Ġsubs Gra usha...
24572    24572  6.527727e-09  0.185225  0.987419  12.197604  -3.828456  Ġintent ĠIntent intent Ġintention Ġintentions ...
24573    24573 -5.497333e-08  0.160370  0.087656   3.220747 -10.000000  picture ilee gee Reviewed Ġre kefeller ĠPatch ...
24574    24574 -1.910878e-08  0.161196  0.014309   3.154507 -10.000000  thy leans ĠLaksh NetMessage GR ivable lean ĠRe...
24575    24575 -4.592180e-09  0.165507 -0.006775   3.192047  -5.010295  >( hest Ġpatiently "}, paralle Fact anthrop al...

[24576 rows x 7 columns]
```

### 1. Interpreting Features

Use GPT4o to interpret the features based on the top activation tokens. The results are written to a text file.

```bash
python3 src/1_interpret_features.py \
  --input_file top_activations_per_feature.json \
  --output_file interpreted_features.json \
  --openai_api_key secretSquirrel
```

or

```bash
python3 src/1_interpret_features.py \
  --input_file $TOP_ACTIVATIONS_FILE \
  --output_file $INTERPRETED_FEATURES_FILE \
  --openai_api_key $OPENAI_API_KEY
```

Expected output:
```
File batch status: completed
File counts: FileCounts(cancelled=0, completed=1, failed=0, in_progress=0, total=1)

assistant > file_search

... LLM response omitted ...

Results saved to interpreted_features.json
```

### 3. Steering Features

Find a feature of interest (from the analyzed file in Step 2) and apply a steering vector that influences the model with respect to the selected feature. Experiment with the temperature and coefficient multiplier to observe behavioral shifts.

```bash
python3 src/2_apply_vector_steering.py \
  --model_name gpt2-small \
  --sae_release gpt2-small-res-jb \
  --sae_id blocks.0.hook_resid_pre \
  --layer 8 \
  --feature_id 11952 \
  --coeff 300 \
  --temperature 1 \
  --prompt "What is your favorite animal?
```

or

```bash
python3 src/2_apply_vector_steering.py \
  --model_name $MODEL_NAME \
  --sae_release $SAE_RELEASE \
  --sae_id $SAE_ID \
  --layer $LAYER \
  --feature_id $FEATURE_TO_STEER \
  --coeff $STEERING_COEFF \
  --temperature $STEERING_TEMP \
  --prompt $STEERING_PROMPT
```

Expected Output:
```
Generation with steering:
100%
 50/50 [00:02<00:00, 34.32it/s]
What is your favorite animal? Pokemon Pokemon Pokemon Pokemon Pokemon typePokemon evolve Pokemon the original is,. ball. was d pokemon that a two I- all n wild, am feel no more p and other or in p at the w to m up j/ po house a two

--------------------------------------------------------------------------------

What is your favorite animal? Pokemon Pokemon Pokemon Pokemon PokemonPokemon am,.
., level I ball wild at a to other and are moves and the s so pokemon- f b h more ch evolve ball d that black k z c ge ev n- t am. m

--------------------------------------------------------------------------------

What is your favorite animal? Pokemon Pokemon Pokemon Pokemon Pokemon Stadium. are is the I,, move am evolve poke pokemon f can level at that other. so a s all. g- like ge wild I do four other b my n k- f ball w are n evolved

Generation without steering:
100%
 50/50 [00:01<00:00, 31.32it/s]
What is your favorite animal?

Lobster Munchies are a family of fast-food hamburgers that have been in the U.S. for over 50 years. They're often referred to as "lobsters" because they're so delicious and full

--------------------------------------------------------------------------------

What is your favorite animal?

The most common answer to this question is "Cocoon." Cocoons are the largest carnivores in the world. They live in almost every part of the world, including Africa, Asia and Europe. Their habitat consists of over 50

--------------------------------------------------------------------------------

What is your favorite animal?

I love dogs, cats, and horses. I also love rabbits and rabbits. My dog is a big furry fan of the circus animals. He loves to play with his friends and even make me laugh! He's always been very happy to
```

Done!

## Current Status

This project is currently in development. Please check the issues and project boards for the most up-to-date status and ongoing tasks.

## Contributing

This is an academic project for DATASCI266. Contributions are limited to the assigned group members and course instructors.

## License

TBD