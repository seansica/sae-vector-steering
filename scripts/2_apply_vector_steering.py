import argparse
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from typing import Dict, List
from transformer_lens.hook_points import HookPoint

from main import get_device


def load_model_and_sae(
    model_name: str, sae_release: str, sae_id: str, layer: int, device: str = "cpu"
) -> tuple[HookedTransformer, SAE]:
    model = HookedTransformer.from_pretrained(model_name).to(device)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Apply vector steering")
    parser.add_argument("--device", help="Manually specify the device from [cuda, mps, cpu]", default=get_device())
    parser.add_argument("--model_name", required=True, help="Name of the Transformer model")
    parser.add_argument("--sae_release", required=True, help="The SAE release name")
    parser.add_argument("--sae_id", required=True, help="The ID of the SAE to load")
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="The SAE layer to extract features from",
    )
    parser.add_argument(
        "--feature_id", type=int, required=True, help="Feature ID to use for steering"
    )
    parser.add_argument(
        "--coeff", type=float, required=True, help="Coefficient for steering"
    )
    parser.add_argument(
        "--temperature", type=float, required=True, help="Temperature for steering"
    )
    parser.add_argument("--prompt", required=True, help="Prompt for steering")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and SAE
    model, sae = load_model_and_sae(
        args.model_name, args.sae_release, args.sae_id, args.layer, args.device
    )

    # Create the steering vector
    steering_vector = sae.W_dec[args.feature_id]

    # Set up generation parameters
    sampling_kwargs = dict(temperature=args.temperature, top_p=0.3, freq_penalty=1.0)

    # Run generation with and without steering
    run_generate(
        model,
        args.prompt,
        f"blocks.{args.layer}.hook_resid_pre",
        steering_vector,
        args.coeff,
        sampling_kwargs,
    )


if __name__ == "__main__":
    main()
