import nnsight
from nnsight import CONFIG
from nnsight import LanguageModel, util
from nnsight.tracing.graph import Proxy

from typing import Tuple, List
import plotly.express as px
import plotly.io as pio

import numpy as np
import torch
import gc
import json
import random
import einops

# Set API key
CONFIG.API.APIKEY = input("enter api key: ")
print("API key set")


def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_counting_dataset(filename="dataset.json"):
    """Load counting examples from JSON file"""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using generated examples.")
        return None


def create_better_counting_examples():
    """
    Create IOI-style clean/corrupted pairs where only ONE WORD differs
    Following the IOI pattern of swapping one key word that changes the answer
    """
    print("\n--- Creating IOI-style counting examples ---")

    # Hardcode examples like IOI does
    clean_examples = []
    corrupted_examples = []
    clean_counts = []
    corrupted_counts = []

    # Example 1: Clean has 2 animals, corrupted has 3 animals (swap table->dog)
    clean_prompt = "Count the animals: fish table cat. Answer:"
    corrupted_prompt = "Count the animals: fish dog cat. Answer:"
    clean_examples.append(clean_prompt)
    corrupted_examples.append(corrupted_prompt)
    clean_counts.append(2)  # fish, cat
    corrupted_counts.append(3)  # fish, dog, cat

    # Example 2: Clean has 1 animal, corrupted has 2 animals (swap book->bird)
    clean_prompt = "Count the animals: dog book pen. Answer:"
    corrupted_prompt = "Count the animals: dog bird pen. Answer:"
    clean_examples.append(clean_prompt)
    corrupted_examples.append(corrupted_prompt)
    clean_counts.append(1)  # dog
    corrupted_counts.append(2)  # dog, bird

    # Example 3: Clean has 0 animals, corrupted has 1 animal (swap table->cat)
    clean_prompt = "Count the animals: book table pen. Answer:"
    corrupted_prompt = "Count the animals: book cat pen. Answer:"
    clean_examples.append(clean_prompt)
    corrupted_examples.append(corrupted_prompt)
    clean_counts.append(0)  # no animals
    corrupted_counts.append(1)  # cat

    for i, (clean, corrupted, clean_count, corrupted_count) in enumerate(
        zip(clean_examples, corrupted_examples, clean_counts, corrupted_counts)
    ):
        print(f"Example {i}:")
        print(f"  Clean ({clean_count}): {clean}")
        print(f"  Corrupted ({corrupted_count}): {corrupted}")
        print()

    return clean_examples, corrupted_examples, clean_counts, corrupted_counts


def get_number_tokens(tokenizer, max_count=10):
    """
    Get token IDs for numbers 0-max_count
    """
    print("\n--- Getting number tokens ---")
    number_tokens = {}

    for i in range(max_count + 1):
        # Try space + number first (common in GPT-2)
        text = f" {i}"
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(tokens) == 1:
            number_tokens[i] = tokens[0]
        else:
            # Try just the number
            text = str(i)
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            number_tokens[i] = tokens[0]

    # Verify tokens
    for num, tok in number_tokens.items():
        decoded = tokenizer.decode([tok])
        print(f"{num} -> token {tok} -> '{decoded}'")

    return number_tokens


def counting_activation_patching_fixed(model, example_idx=0):
    """
    Fixed activation patching implementation following IOI pattern exactly
    """
    print(
        f"\n--- Running FIXED counting_activation_patching for example {example_idx} ---"
    )
    clear_memory()

    # Get examples
    clean_examples, corrupted_examples, clean_counts, corrupted_counts = (
        create_better_counting_examples()
    )

    if example_idx >= len(clean_examples):
        example_idx = 0

    number_tokens = get_number_tokens(model.tokenizer)

    clean_prompt = clean_examples[example_idx]
    corrupted_prompt = corrupted_examples[example_idx]
    clean_count = clean_counts[example_idx]
    corrupted_count = corrupted_counts[example_idx]

    print(f"Clean prompt: {clean_prompt}")
    print(f"Corrupted prompt: {corrupted_prompt}")
    print(f"Clean expected count: {clean_count}")
    print(f"Corrupted expected count: {corrupted_count}")

    # Set up tokens - like IOI, we compare correct vs incorrect for each prompt
    clean_correct_index = number_tokens[clean_count]
    clean_incorrect_index = number_tokens[
        corrupted_count
    ]  # The corrupted count is "incorrect" for clean

    print(f"Clean correct token ({clean_count}): {clean_correct_index}")
    print(f"Clean incorrect token ({corrupted_count}): {clean_incorrect_index}")

    N_LAYERS = len(model.transformer.h)

    # Get sequence length from clean prompt
    clean_tokens = model.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0]
    seq_len = len(clean_tokens)

    print(f"Sequence length: {seq_len}")

    # Store results
    patching_results = []

    # Run all forward passes in one tracing context
    with model.trace() as tracer:

        # 1. Clean run - save all layer outputs
        with tracer.invoke(clean_prompt) as invoker:
            clean_layer_outputs = {}
            for layer_idx in range(N_LAYERS):
                clean_layer_outputs[layer_idx] = model.transformer.h[layer_idx].output[
                    0
                ]

            clean_logits = model.lm_head.output
            clean_logit_diff = (
                clean_logits[0, -1, clean_correct_index]
                - clean_logits[0, -1, clean_incorrect_index]
            ).save()

        # 2. Corrupted run - baseline
        with tracer.invoke(corrupted_prompt) as invoker:
            corrupted_logits = model.lm_head.output
            corrupted_logit_diff = (
                corrupted_logits[0, -1, clean_correct_index]
                - corrupted_logits[0, -1, clean_incorrect_index]
            ).save()

        # 3. Patched runs - for each layer and position
        for layer_idx in range(N_LAYERS):
            layer_results = []

            for pos_idx in range(seq_len):
                # Patched run: corrupted prompt but with clean activations at layer_idx, position pos_idx
                with tracer.invoke(corrupted_prompt) as invoker:
                    # Replace the activation at this layer and position with clean version
                    model.transformer.h[layer_idx].output[0][:, pos_idx, :] = (
                        clean_layer_outputs[layer_idx][:, pos_idx, :]
                    )

                    patched_logits = model.lm_head.output
                    patched_logit_diff = (
                        patched_logits[0, -1, clean_correct_index]
                        - patched_logits[0, -1, clean_incorrect_index]
                    )

                    # Calculate restoration: how much did patching help?
                    # This measures: (patched - corrupted) / (clean - corrupted)
                    restoration = (patched_logit_diff - corrupted_logit_diff) / (
                        clean_logit_diff - corrupted_logit_diff
                    )

                    layer_results.append(restoration.save())

            patching_results.append(layer_results)
            print(f"Completed layer {layer_idx}")

    # Get token labels
    tokens = model.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0]
    token_labels = [
        f"{model.tokenizer.decode(token)}_{i}" for i, token in enumerate(tokens)
    ]

    return (
        patching_results,
        token_labels,
        float(clean_logit_diff),
        float(corrupted_logit_diff),
    )


def plot_ioi_patching_results(
    model,
    ioi_patching_results,
    x_labels,
    plot_title="Normalized Logit Difference After Patching Residual Stream on the Counting Task",
):
    """
    Plot patching results with proper conversion
    """
    # Convert saved tensors to values
    results_array = []
    for layer_results in ioi_patching_results:
        layer_values = [float(result.value) for result in layer_results]
        results_array.append(layer_values)

    results_array = np.array(results_array)

    # Check dimensions
    print(f"Results shape: {results_array.shape}")
    print(f"Labels length: {len(x_labels)}")

    # Ensure the number of columns matches the number of labels
    if results_array.shape[1] != len(x_labels):
        # Trim results to match labels or vice versa
        min_length = min(results_array.shape[1], len(x_labels))
        results_array = results_array[:, :min_length]
        x_labels = x_labels[:min_length]
        print(f"Adjusted to {min_length} tokens")

    fig = px.imshow(
        results_array,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Position", "y": "Layer", "color": "Norm. Logit Diff"},
        x=x_labels,
        y=[f"Layer {i}" for i in range(results_array.shape[0])],
        title=plot_title,
    )

    # Save plot
    try:
        fig.write_image("counting_results.png")
    except:
        print("Could not save image (kaleido not installed)")

    return fig


def counting_activation_patching_llama(model, example_idx=0):
    """
    Activation patching for counting task using Llama-style model architecture
    Patches attention head outputs instead of residual stream
    """
    print(
        f"\n--- Running Llama counting_activation_patching for example {example_idx} ---"
    )
    clear_memory()

    # Get examples
    clean_examples, corrupted_examples, clean_counts, corrupted_counts = (
        create_better_counting_examples()
    )

    if example_idx >= len(clean_examples):
        example_idx = 0

    number_tokens = get_number_tokens(model.tokenizer)

    clean_prompt = clean_examples[example_idx]
    corrupted_prompt = corrupted_examples[example_idx]
    clean_count = clean_counts[example_idx]
    corrupted_count = corrupted_counts[example_idx]

    print(f"Clean prompt: {clean_prompt}")
    print(f"Corrupted prompt: {corrupted_prompt}")
    print(f"Clean expected count: {clean_count}")
    print(f"Corrupted expected count: {corrupted_count}")

    # Set up tokens
    clean_correct_index = number_tokens[clean_count]
    clean_incorrect_index = number_tokens[corrupted_count]

    print(f"Clean correct token ({clean_count}): {clean_correct_index}")
    print(f"Clean incorrect token ({corrupted_count}): {clean_incorrect_index}")

    # Model architecture parameters
    N_LAYERS = len(model.model.layers)
    N_HEADS = model.config.num_attention_heads  # Get from config
    D_MODEL = model.config.hidden_size
    D_HEADS = D_MODEL // N_HEADS

    print(f"Model: {N_LAYERS} layers, {N_HEADS} heads, {D_MODEL} hidden size")

    # Store results
    patching_results = []

    # Run all forward passes in one tracing context with remote=True for larger models
    with model.trace(remote=True) as tracer:

        # 1. Clean run - save attention head outputs
        with tracer.invoke(clean_prompt) as invoker:
            clean_tokens = invoker.inputs[0][0]["input_ids"][0]

            # Store attention head outputs for patching
            z_hs = {}
            for layer_idx in range(N_LAYERS):
                # Get attention output (before o_proj)
                z = model.model.layers[layer_idx].self_attn.o_proj.input
                # Reshape to separate heads: [batch, seq, heads, head_dim]
                z_reshaped = einops.rearrange(z, "b s (nh dh) -> b s nh dh", nh=N_HEADS)
                for head_idx in range(N_HEADS):
                    z_hs[layer_idx, head_idx] = z_reshaped[:, :, head_idx, :]

            clean_logits = model.lm_head.output
            clean_logit_diff = (
                clean_logits[0, -1, clean_correct_index]
                - clean_logits[0, -1, clean_incorrect_index]
            ).save()

        # 2. Corrupted run - baseline
        with tracer.invoke(corrupted_prompt) as invoker:
            corrupted_logits = model.lm_head.output
            corrupted_logit_diff = (
                corrupted_logits[0, -1, clean_correct_index]
                - corrupted_logits[0, -1, clean_incorrect_index]
            ).save()

        # 3. Patched runs - for each layer and attention head
        for layer_idx in range(N_LAYERS):
            layer_results = []

            for head_idx in range(N_HEADS):
                # Patched run: corrupted prompt but with clean attention head output
                with tracer.invoke(corrupted_prompt) as invoker:
                    # Get corrupted attention output
                    z_corrupt = model.model.layers[layer_idx].self_attn.o_proj.input
                    # Reshape to separate heads
                    z_corrupt = einops.rearrange(
                        z_corrupt, "b s (nh dh) -> b s nh dh", nh=N_HEADS
                    )
                    # Patch this specific head with clean activation
                    z_corrupt[:, :, head_idx, :] = z_hs[layer_idx, head_idx]
                    # Reshape back
                    z_corrupt = einops.rearrange(
                        z_corrupt, "b s nh dh -> b s (nh dh)", nh=N_HEADS
                    )
                    # Apply the patch
                    model.model.layers[layer_idx].self_attn.o_proj.input = z_corrupt

                    patched_logits = model.lm_head.output
                    patched_logit_diff = (
                        patched_logits[0, -1, clean_correct_index]
                        - patched_logits[0, -1, clean_incorrect_index]
                    )

                    # Calculate restoration
                    restoration = (patched_logit_diff - corrupted_logit_diff) / (
                        clean_logit_diff - corrupted_logit_diff
                    )

                    layer_results.append(restoration.save())

            patching_results.append(layer_results)
            print(f"Completed layer {layer_idx}")

    # Create labels for attention heads
    head_labels = [f"Head_{i}" for i in range(N_HEADS)]

    return (
        patching_results,
        head_labels,
        float(clean_logit_diff),
        float(corrupted_logit_diff),
    )


def test_model_counting_ability_llama(model):
    """
    Test if the Llama model can actually count animals reliably
    """
    print("\n--- Testing Llama Model Counting Ability ---")

    test_prompts = [
        ("Count the animals: fish table cat. Answer:", 2),
        ("Count the animals: fish dog cat. Answer:", 3),
        ("Count the animals: dog book pen. Answer:", 1),
        ("Count the animals: dog bird pen. Answer:", 2),
        ("Count the animals: book table pen. Answer:", 0),
        ("Count the animals: book cat pen. Answer:", 1),
        ("Count the animals: cat dog bird fish horse. Answer:", 5),
        ("Count the animals: table chair book pen car. Answer:", 0),
    ]

    number_tokens = get_number_tokens(model.tokenizer, max_count=5)

    correct_predictions = 0
    total_predictions = len(test_prompts)

    with model.trace(remote=True) as tracer:
        for prompt, expected_count in test_prompts:
            with tracer.invoke(prompt) as invoker:
                logits = model.lm_head.output[0, -1, :]

                # Get probabilities for numbers 0-5
                number_probs = {}
                for num in range(6):
                    if num in number_tokens:
                        token_id = number_tokens[num]
                        number_probs[num] = float(
                            torch.softmax(logits, dim=-1)[token_id]
                        )

                # Find predicted number
                predicted_count = max(
                    number_probs.keys(), key=lambda x: number_probs[x]
                )

                # Check if correct
                is_correct = predicted_count == expected_count
                if is_correct:
                    correct_predictions += 1

                print(f"Prompt: {prompt}")
                print(
                    f"Expected: {expected_count}, Predicted: {predicted_count} {'✓' if is_correct else '✗'}"
                )
                print(
                    f"Number probabilities: {[(num, f'{prob:.3f}') for num, prob in sorted(number_probs.items())]}"
                )
                print()

    accuracy = correct_predictions / total_predictions
    print(
        f"Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})"
    )

    if accuracy < 0.5:
        print(
            "⚠️  Model performs poorly on counting - activation patching may not work well"
        )
    elif accuracy < 0.8:
        print("⚠️  Model has moderate counting ability - expect weak patching signals")
    else:
        print("✓ Model has good counting ability - activation patching should work")

    return accuracy


def plot_attention_head_patching_results(
    model,
    patching_results,
    head_labels,
    plot_title="Attention Head Patching Results on Counting Task",
):
    """
    Plot attention head patching results
    """
    # Convert saved tensors to values
    results_array = []
    for layer_results in patching_results:
        layer_values = [float(result.value) for result in layer_results]
        results_array.append(layer_values)

    results_array = np.array(results_array)

    print(f"Results shape: {results_array.shape}")
    print(f"Labels length: {len(head_labels)}")

    fig = px.imshow(
        results_array,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Attention Head", "y": "Layer", "color": "Restoration"},
        x=head_labels,
        y=[f"Layer {i}" for i in range(results_array.shape[0])],
        title=plot_title,
    )

    # Save plot
    try:
        fig.write_image("counting_attention_results.png")
    except:
        print("Could not save image (kaleido not installed)")

    return fig


def analyze_layer_importance_fixed(patching_results, token_labels):
    """
    Analyze results with proper conversion
    """
    # Convert saved tensors to values
    results_array = []
    for layer_results in patching_results:
        layer_values = [float(result.value) for result in layer_results]
        results_array.append(layer_values)

    results_array = np.array(results_array)

    print(f"Results shape: {results_array.shape}")
    print(f"Results range: [{results_array.min():.4f}, {results_array.max():.4f}]")

    # Find max importance per layer
    max_per_layer = np.max(np.abs(results_array), axis=1)

    print("\n--- Layer Importance Analysis ---")
    print("Max absolute restoration per layer:")
    for i, val in enumerate(max_per_layer):
        print(f"  Layer {i}: {val:.4f}")

    print("\nTop 5 most important layers:")
    top_layers = np.argsort(max_per_layer)[-5:][::-1]
    for layer_idx in top_layers:
        pos_idx = np.argmax(np.abs(results_array[layer_idx]))
        print(
            f"  Layer {layer_idx}: {max_per_layer[layer_idx]:.4f} (at position {pos_idx}: {token_labels[pos_idx]})"
        )

    # Find average importance per position
    avg_per_position = np.mean(np.abs(results_array), axis=0)

    print("\nAverage importance per token position:")
    # Show first 10 tokens
    for i in range(min(10, len(token_labels))):
        print(f"  {token_labels[i]}: {avg_per_position[i]:.4f}")
    if len(token_labels) > 10:
        print(f"  ... ({len(token_labels) - 10} more tokens)")

    return results_array


def analyze_attention_head_importance(patching_results, head_labels):
    """
    Analyze which attention heads are most important for counting
    """
    # Convert saved tensors to values
    results_array = []
    for layer_results in patching_results:
        layer_values = [float(result.value) for result in layer_results]
        results_array.append(layer_values)

    results_array = np.array(results_array)

    print(f"Results shape: {results_array.shape}")
    print(f"Results range: [{results_array.min():.4f}, {results_array.max():.4f}]")

    # Find max importance per layer
    max_per_layer = np.max(np.abs(results_array), axis=1)

    print("\n--- Attention Head Importance Analysis ---")
    print("Max absolute restoration per layer:")
    for i, val in enumerate(max_per_layer):
        print(f"  Layer {i}: {val:.4f}")

    print("\nTop 5 most important layers:")
    top_layers = np.argsort(max_per_layer)[-5:][::-1]
    for layer_idx in top_layers:
        head_idx = np.argmax(np.abs(results_array[layer_idx]))
        print(
            f"  Layer {layer_idx}: {max_per_layer[layer_idx]:.4f} (at {head_labels[head_idx]})"
        )

    # Find most important attention heads across all layers
    flat_results = results_array.flatten()
    flat_indices = np.argsort(np.abs(flat_results))[-10:][::-1]  # Top 10

    print("\nTop 10 most important attention heads:")
    for idx in flat_indices:
        layer_idx = idx // len(head_labels)
        head_idx = idx % len(head_labels)
        value = flat_results[idx]
        print(f"  Layer {layer_idx}, {head_labels[head_idx]}: {value:.4f}")

    return results_array


def run_llama_counting_analysis(model_name="meta-llama/Meta-Llama-3.1-8B"):
    """
    Complete pipeline for analyzing counting mechanisms in Llama models
    """
    print("--- Starting Llama Counting Analysis ---")
    print(f"Using model: {model_name}")

    print(f"Loading model...")
    try:
        model = LanguageModel(model_name)
        print(f"Model loaded successfully")
        print(f"Model config: {model.config}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    clear_memory()

    # Test counting ability first
    accuracy = test_model_counting_ability_llama(model)

    if accuracy < 0.3:
        print(
            f"\n❌ Model accuracy too low ({accuracy:.2%}) - skipping activation patching"
        )
        return None

    print(f"\n✓ Proceeding with attention head patching (accuracy: {accuracy:.2%})")

    # Run attention head patching
    all_results = []
    for example_idx in range(3):
        print(f"\n{'='*60}")
        print(f"Testing example {example_idx}")
        print(f"{'='*60}")

        try:
            patching_results, head_labels, clean_diff, corrupted_diff = (
                counting_activation_patching_llama(model, example_idx=example_idx)
            )

            print(f"Clean logit difference: {clean_diff:.4f}")
            print(f"Corrupted logit difference: {corrupted_diff:.4f}")
            print(f"Difference magnitude: {abs(clean_diff - corrupted_diff):.4f}")

            fig = plot_attention_head_patching_results(
                model,
                patching_results,
                head_labels,
                plot_title=f"Patching Llama Attention Heads on Counting Task (Example {example_idx})",
            )

            try:
                pio.renderers.default = "browser"
                fig.show()
            except:
                print(
                    "Could not display plot in browser, saved to counting_attention_results.png"
                )

            analyze_attention_head_importance(patching_results, head_labels)
            all_results.append((patching_results, head_labels))

        except Exception as e:
            print(f"Error processing example {example_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    return all_results


def run_counting_analysis(model_name="openai-community/gpt2", force_cpu=False):
    """
    Complete pipeline for analyzing counting mechanisms
    """
    print("--- Starting run_counting_analysis ---")
    print(f"Using model: {model_name}")

    # Check available memory
    device = "cpu" if force_cpu else "auto"

    if not force_cpu and torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            free_memory = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)
            ) / 1e9
            print(f"GPU total memory: {gpu_memory:.2f} GB")
            print(f"GPU free memory: {free_memory:.2f} GB")

            if free_memory < 2:  # If less than 2GB free, use CPU
                device = "cpu"
                print("Low GPU memory detected, switching to CPU")
        except:
            print("Could not check GPU memory, using auto device placement")

    print(f"Loading model with device: {device}...")

    # Load model with appropriate device
    try:
        if device == "cpu":
            model = LanguageModel(model_name, device_map="cpu")
        else:
            model = LanguageModel(model_name, device_map="auto")
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model on {device}, trying CPU: {e}")
        model = LanguageModel(model_name, device_map="cpu")
        print("Model loaded on CPU")

    clear_memory()

    # Test multiple examples
    all_results = []
    for example_idx in range(3):  # Test first 3 examples
        print(f"\n{'='*60}")
        print(f"Testing example {example_idx}")
        print(f"{'='*60}")

        try:
            ioi_patching_results, token_labels, clean_diff, corrupted_diff = (
                counting_activation_patching_fixed(model, example_idx=example_idx)
            )

            # Always plot and analyze - logit differences for counting might be small
            print(f"Clean logit difference: {clean_diff:.4f}")
            print(f"Corrupted logit difference: {corrupted_diff:.4f}")
            print(f"Difference magnitude: {abs(clean_diff - corrupted_diff):.4f}")

            fig = plot_ioi_patching_results(
                model,
                ioi_patching_results,
                token_labels,
                plot_title=f"Patching GPT-2 Residual Stream on Counting Task (Example {example_idx})",
            )

            # Show plot
            try:
                pio.renderers.default = "browser"
                fig.show()
            except:
                print(
                    "Could not display plot in browser, saved to counting_results.png"
                )

            # Analyze importance
            analyze_layer_importance_fixed(ioi_patching_results, token_labels)

            all_results.append((ioi_patching_results, token_labels))

        except torch.cuda.OutOfMemoryError as e:
            print(
                f"CUDA OOM on example {example_idx}, clearing memory and retrying on CPU"
            )
            clear_memory()

            # Force CPU for remaining examples
            if not force_cpu:
                print("Restarting analysis on CPU...")
                return run_counting_analysis(model_name=model_name, force_cpu=True)
            else:
                print(f"Still getting OOM on CPU, skipping example {example_idx}")
                continue

        except Exception as e:
            print(f"Error processing example {example_idx}: {e}")
            import traceback

            traceback.print_exc()
            clear_memory()
            continue

    return all_results


if __name__ == "__main__":
    print("--- Script execution started ---")

    # Choose which model to run
    model_choice = input(
        "Choose model: (1) GPT-2 small (2) Llama-3.1-8B [default: 1]: "
    ).strip()

    try:
        if model_choice == "2":
            print("Running Llama analysis...")
            results = run_llama_counting_analysis()
        else:
            print("Running GPT-2 analysis...")
            results = run_counting_analysis_with_ability_test()

        print("\n--- Script execution finished successfully ---")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback

        traceback.print_exc()
        clear_memory()
        raise
