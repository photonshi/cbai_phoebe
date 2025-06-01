# for a single model, create a causal mediation analysis experiment 
# (patching from one run to another) to answer: 
# "is there a hidden state layer that contains a representation of the running count of matching words, 
# while processing the list of words?"

import nnsight
from nnsight import CONFIG
from nnsight import LanguageModel, util
from nnsight.tracing.graph import Proxy

from typing import Tuple
import plotly.express as px
import plotly.io as pio

import numpy as np

from benchmark import generate_question

from categories import *

# 7c00f2a8-9c9c-4651-bcf0-df3b6f4b88b8
CONFIG.API.APIKEY = input("enter api key")
print("API key set")

def create_counting_examples():
    """
    Create clean/corrupted pairs for counting task - Updated to encourage numerical output
    """
    print("\n--- Running create_counting_examples ---")
    # Clean examples with animals to count - emphasize numerical output
    clean_examples = [
        "Count the animals and give the answer as a number: dog apple cherry bus cat grape bowl. Answer:",
        "Count the animals and give the answer as a number: bird table lion phone duck water glass. Answer:", 
        "Count the animals and give the answer as a number: mouse book tiger pen fish soup knife. Answer:",
        "Count the animals in this list and give the answer as a number: cat dog bird lion mouse fish tiger. Answer:",
        "How many animals are in this list? Give the answer as a number: elephant car tree rabbit book horse desk. Answer:"
    ]
    
    # Corrupted examples - replace animals with non-animals
    corrupted_examples = [
        "Count the animals and give the answer as a number: book apple cherry bus phone grape bowl. Answer:",
        "Count the animals and give the answer as a number: phone table car phone desk water glass. Answer:",
        "Count the animals and give the answer as a number: book book car pen soup soup knife. Answer:", 
        "Count the animals in this list and give the answer as a number: car phone book car book soup desk. Answer:",
        "How many animals are in this list? Give the answer as a number: phone car tree book book desk desk. Answer:"
    ]
    
    # Expected counts for clean examples
    clean_counts = [2, 3, 3, 7, 3]  # [dog, cat], [bird, lion, duck], [mouse, tiger, fish], etc.
    
    # Expected counts for corrupted examples (should be 0 or very low)
    corrupted_counts = [0, 0, 0, 0, 0]
    
    print(f"Generated {len(clean_examples)} clean examples and {len(corrupted_examples)} corrupted examples.")
    return clean_examples, corrupted_examples, clean_counts, corrupted_counts

def get_number_tokens(tokenizer, max_count=10):
    """
    Get token IDs for numbers 0-max_count - Fixed for GPT-2 tokenizer
    """
    print("\n--- Running get_number_tokens ---")
    number_tokens = {}
    
    for i in range(max_count + 1):
        # Focus on numerical representations: " 2", "2" (prioritize digits over words)
        candidates = [f" {i}", str(i)]
        
        for candidate in candidates:
            try:
                # Handle GPT-2 tokenizer Encoding objects
                token_result = tokenizer(candidate)
                
                # Extract token IDs from Encoding object
                if hasattr(token_result, 'ids'):
                    token_ids = token_result.ids
                elif isinstance(token_result, dict) and "input_ids" in token_result:
                    token_ids = token_result["input_ids"]
                elif hasattr(token_result, 'input_ids'):
                    token_ids = token_result.input_ids
                else:
                    token_ids = token_result
                
                # Skip BOS token if present, handle single token case
                if len(token_ids) >= 2:
                    number_tokens[i] = token_ids[1]
                    break
                elif len(token_ids) == 1:
                    number_tokens[i] = token_ids[0]
                    break
            except Exception as e:
                print(f"Error tokenizing '{candidate}': {e}")
                continue
                
        if i not in number_tokens:
            # Fallback - just use string representation
            try:
                token_result = tokenizer(str(i))
                if hasattr(token_result, 'ids'):
                    token_ids = token_result.ids
                else:
                    token_ids = [0]  # Emergency fallback
                number_tokens[i] = token_ids[0] if len(token_ids) > 0 else 0
            except:
                number_tokens[i] = 0  # Emergency fallback
    
    print(f"Number to token mapping: {number_tokens}")
    return number_tokens

def counting_activation_patching(llm, example_idx=0):
    """
    Apply activation patching to understand counting representations - Fixed for GPT-2
    """
    print(f"\n--- Running counting_activation_patching for example_idx: {example_idx} ---")
    clean_examples, corrupted_examples, clean_counts, corrupted_counts = create_counting_examples()
    number_tokens = get_number_tokens(llm.tokenizer)
    
    clean_prompt = clean_examples[example_idx]
    corrupted_prompt = corrupted_examples[example_idx]
    correct_count = clean_counts[example_idx]
    
    print(f"Clean prompt: {clean_prompt}")
    print(f"Corrupted prompt: {corrupted_prompt}")
    print(f"Expected count: {correct_count}")
    
    # Get token IDs for the correct count and a wrong count
    correct_token = number_tokens[correct_count]
    wrong_token = number_tokens[0]  # "0" as wrong answer
    
    print(f"Correct token ID ({correct_count}): {correct_token}")
    print(f"Wrong token ID (0): {wrong_token}")
    
    # GPT-2 specific architecture - Fixed model path
    N_LAYERS = len(llm.transformer.h)  # GPT-2 uses transformer.h, not model.transformer.h
    print(f"Model has {N_LAYERS} layers")
    
    patching_results = []
    
    with llm.trace(remote=True) as tracer:
        # Clean run - save activations
        print("\nStarting clean run...")
        with tracer.invoke(clean_prompt) as invoker:
            # Handle tokenizer input properly
            # The invoker.inputs structure for a LanguageModel.invoke('string_prompt') is:
            # (( {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])} ), {})
            # So invoker.inputs[0] is the tuple containing the tokenized dict.
            # invoker.inputs[0][0] is the tokenized dict itself.
            # invoker.inputs[0][0]['input_ids'] is the batched input_ids tensor.
            # We take the first batch element [0].
            tokenized_dict = invoker.inputs[0][0]
            clean_tokens = tokenized_dict['input_ids'][0]
            
            # Save residual stream activations after each layer (GPT-2 specific)
            clean_residual = {}
            for layer_idx in range(N_LAYERS):
                # For GPT-2: get the residual stream after each transformer block
                layer_block = llm.transformer.h[layer_idx]
                # The residual stream flows through each block. Access the hidden_states (usually the first element).
                clean_residual[layer_idx] = layer_block.output[0]
            
            # Get clean logits
            clean_logits = llm.lm_head.output
            clean_logit_diff = (
                clean_logits[0, -1, correct_token] - clean_logits[0, -1, wrong_token]
            ).save()
        
        # Corrupted run - get baseline performance
        print("\nStarting corrupted run...")
        with tracer.invoke(corrupted_prompt) as invoker:
            corrupted_logits = llm.lm_head.output
            corrupted_logit_diff = (
                corrupted_logits[0, -1, correct_token] - corrupted_logits[0, -1, wrong_token]
            ).save()
        
        # Patching experiments - test each layer's contribution
        print("\nStarting patching experiments...")
        for layer_idx in range(N_LAYERS):
            layer_results = []
            print(f"  Patching Layer {layer_idx}/{N_LAYERS - 1}...")
            
            # Patch at each token position to see when counting representations develop
            token_count = len(clean_tokens) if hasattr(clean_tokens, '__len__') else 10  # fallback
            
            for token_idx in range(token_count):
                with tracer.invoke(corrupted_prompt) as invoker:
                    layer_block = llm.transformer.h[layer_idx]
                    # Patch the hidden_states (first element of output) at this layer and position
                    layer_block.output[0][:, token_idx, :] = clean_residual[layer_idx][:, token_idx, :]
                    
                    patched_logits = llm.lm_head.output
                    patched_logit_diff = (
                        patched_logits[0, -1, correct_token] - patched_logits[0, -1, wrong_token]
                    )
                    
                    # Normalized patching score
                    patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                        clean_logit_diff - corrupted_logit_diff
                    )
                    
                    layer_results.append(patched_result.save())
            
            patching_results.append(layer_results)
    
    # Access .value only after the trace is complete
    print(f"Clean logit diff: {clean_logit_diff.value if hasattr(clean_logit_diff, 'value') else 'pending computation'}")
    print(f"Corrupted logit diff: {corrupted_logit_diff.value if hasattr(corrupted_logit_diff, 'value') else 'pending computation'}")

    return patching_results, clean_tokens, clean_logit_diff, corrupted_logit_diff

def analyze_counting_layers(llm, tokenizer, num_examples=2):
    """
    Run patching on multiple examples and identify which layers are most important for counting
    """
    print("\n--- Running analyze_counting_layers ---")
    all_results = []
    token_labels = []
    
    for example_idx in range(num_examples):
        print(f"\n=== Analyzing Example {example_idx + 1}/{num_examples} ===")
        results, tokens, clean_diff, corrupt_diff = counting_activation_patching(llm, example_idx)
        
        # Convert tokens to readable labels
        if example_idx == 0:  # Only need to do this once
            if hasattr(tokens, '__iter__'):
                token_labels = []
                for token in tokens:
                    # Handle tensor tokens
                    if hasattr(token, 'item'):
                        token_val = token.item()
                    else:
                        token_val = token
                    # Decode single token
                    decoded = tokenizer.decode([token_val])
                    token_labels.append(decoded)
            else:
                token_labels = [f"token_{i}" for i in range(10)]  # fallback
        
        all_results.append(results)
    
    # Average results across examples
    # Convert Proxy objects to values
    processed_results = []
    for example in all_results:
        example_values = []
        for layer in example:
            layer_values = []
            for result in layer:
                if hasattr(result, 'value'):
                    layer_values.append(result.value)
                elif hasattr(result, 'item'):
                    layer_values.append(result.item())
                else:
                    layer_values.append(float(result) if result is not None else 0.0)
            example_values.append(layer_values)
        processed_results.append(example_values)
    
    averaged_results = np.mean(processed_results, axis=0)
    
    return averaged_results, token_labels

def plot_counting_analysis(results, token_labels, title="Counting Representation Development"):
    """
    Visualize where counting representations develop in the model
    """
    fig = px.imshow(
        results,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Token Position", "y": "Layer", "color": "Patching Effect"},
        x=token_labels,
        title=title,
        aspect="auto"
    )
    
    # Add vertical lines to separate different parts of the prompt
    fig.update_layout(
        xaxis=dict(tickangle=45),
        height=600,
        width=1000
    )
    
    return fig

def identify_counting_circuits(results, token_labels, threshold=0.3):
    """
    Identify which (layer, position) pairs are most important for counting
    """
    important_positions = []
    
    for layer_idx, layer_results in enumerate(results):
        for token_idx, effect in enumerate(layer_results):
            if abs(effect) > threshold:
                token_label = token_labels[token_idx] if token_idx < len(token_labels) else f"token_{token_idx}"
                important_positions.append({
                    'layer': layer_idx,
                    'position': token_idx,
                    'token': token_label,
                    'effect': effect,
                    'description': f"Layer {layer_idx} processes '{token_label}' (effect: {effect:.3f})"
                })
    
    # Sort by effect magnitude
    important_positions.sort(key=lambda x: abs(x['effect']), reverse=True)
    
    print("\nMost Important Counting Circuit Components:")
    print("=" * 50)
    for pos in important_positions[:10]:  # Top 10
        print(pos['description'])
    
    return important_positions

def run_counting_analysis(model_name="openai-community/gpt2"):
    """
    Complete pipeline for analyzing counting mechanisms
    """
    print("--- Starting run_counting_analysis ---")
    print(f"Using model: {model_name}")
    print("Loading model...")
    
    llm = LanguageModel(model_name, device_map="auto")
    print(f"Model loaded successfully: {llm}")
    
    print("\nRunning activation patching analysis...")
    results, token_labels = analyze_counting_layers(llm, llm.tokenizer, num_examples=2)
    
    print("\nGenerating visualizations...")
    fig = plot_counting_analysis(results, token_labels)
    if fig:
        fig.show()
    
    print("\nIdentifying important circuit components...")
    circuits = identify_counting_circuits(results, token_labels)
    
    print("--- Finished run_counting_analysis ---")
    return results, token_labels, circuits

if __name__ == "__main__":
    print("--- Script execution started ---")
    results, labels, circuits = run_counting_analysis()
    print("--- Script execution finished ---")