# Experiments

Experiment results are stored here.

This is where we log the results of each vector steering test (which we refer to as our feature "smoke test").

The `baseline` folder stores experiment results conducted against the following models:
```python
model = HookedTransformer.from_pretrained("gpt2-small")
sae, cfg_dict, sparsity = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre")
```

The `fine-tuned` folder stores experiment results conducted against our fine-tuned model:
```python
# Load the fine-tuned gpt2 model
!unzip sae-vector-steering/models/sae/results_gpt2_medical_no_lora.zip
path_to_state_dict = "fine_tuned_gpt2/model.safetensors"  # Local

from transformer_lens import HookedTransformer
from safetensors import safe_open
import torch

# Initialize the HookedTransformer with the same architecture as your fine-tuned model
model = HookedTransformer.from_pretrained("gpt2")  # base model
# Load the state dict from the .safetensors file
with safe_open(path_to_state_dict, framework="pt", device=device) as f:
    state_dict = {key: f.get_tensor(key) for key in f.keys()}

# Load the state dict into the model
model.load_state_dict(state_dict, strict=False)
model = model.to(device)

# Load the custom trained SAE
!unzip models/sae/sae_custom3.zip
path_to_custom_sae_dict = 'content/checkpoints/h3lkdunu/final_61440000'
sae = SAE.load_from_pretrained(path_to_custom_sae_dict, device=device)
```