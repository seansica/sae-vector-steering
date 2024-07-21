import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from typing import Dict, List
from transformer_lens.hook_points import HookPoint


def load_model_and_sae(
    model_name: str, sae_release: str, sae_id: str, layer: int, device: str = "cpu"
) -> tuple[HookedTransformer, SAE]:
    model = HookedTransformer.from_pretrained(model_name)
    sae, _, _ = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    return model, sae


def steering_hook(
    resid_post: torch.Tensor,
    hook: HookPoint,
    steering_vector: torch.Tensor,
    coeff: float,
):
    if resid_post.shape[1] > 1:
        resid_post[:, :-1, :] += coeff * steering_vector


def hooked_generate(
    model: HookedTransformer,
    prompt_batch: List[str],
    fwd_hooks: List,
    seed: int = None,
    **kwargs,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            stop_at_eos=False,
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs,
        )
    return result


def run_generate(
    model: HookedTransformer,
    example_prompt: str,
    hook_point: str,
    steering_vector: torch.Tensor,
    coeff: float,
    sampling_kwargs: Dict,
):
    model.reset_hooks()

    # Generation with steering
    print("Generation with steering:")
    editing_hooks = [
        (
            hook_point,
            lambda resid_post, hook: steering_hook(
                resid_post, hook, steering_vector, coeff
            ),
        )
    ]
    res = hooked_generate(model, [example_prompt] * 3, editing_hooks, **sampling_kwargs)
    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))

    # Generation without steering
    print("\nGeneration without steering:")
    res = hooked_generate(model, [example_prompt] * 3, [], **sampling_kwargs)
    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))


def get_user_feature_choice(max_feature_id: int) -> int:
    while True:
        try:
            feature_id = int(
                input(
                    f"Enter the feature ID you want to use for steering (0-{max_feature_id}): "
                )
            )
            if 0 <= feature_id <= max_feature_id:
                return feature_id
            else:
                print(
                    f"Please enter a valid feature ID between 0 and {max_feature_id}."
                )
        except ValueError:
            print("Please enter a valid integer.")


def main():
    # Load model and SAE
    model_name = "gpt2-small"
    sae_release = "gpt2-small-res-jb"
    sae_id = "blocks.0.hook_resid_pre"
    layer = 8
    model, sae = load_model_and_sae(model_name, sae_release, sae_id, layer)

    # Get user input for feature ID
    max_feature_id = sae.W_dec.shape[0] - 1
    chosen_feature_id = get_user_feature_choice(max_feature_id)

    # Create the steering vector
    steering_vector = sae.W_dec[chosen_feature_id]

    # Set up generation parameters
    example_prompt = "What is your favorite animal?"
    coeff = 300  # You may need to adjust this
    sampling_kwargs = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)

    # Run generation with and without steering
    run_generate(
        model,
        example_prompt,
        f"blocks.{layer}.hook_resid_pre",
        steering_vector,
        coeff,
        sampling_kwargs,
    )


if __name__ == "__main__":
    main()
