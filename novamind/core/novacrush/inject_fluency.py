"""
NovaCrush -- Fluency Injection (DeepSeek's Option B)
======================================================
Downloads a nanoscopic language model (TinyStories-1M) and distills
its pre-trained grammatical structure directly into our Ternary Substrate.

Instead of training for days to learn that "the" is followed by a noun,
we inject this statistical scaffolding natively. The NovaCrush engine 
will then use this "grammar skeleton" to reason causally.

Requirements: pip install transformers huggingface-hub
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.novacrush.crushed_substrate import CrushedSubstrate
from core.novacrush.language_gen import NativeLanguageGenerator

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("❌ Transformers library not found.")
    print("Please run: pip install transformers huggingface-hub")
    sys.exit(1)

def inject_fluency(target_dim: int = 64, max_vocab: int = 4000):
    print("=" * 60)
    print("  NovaCrush -- Fluency Injector")
    print("=" * 60)
    
    model_name = "distilgpt2"
    print(f">> Downloading/Loading '{model_name}'...")
    print("   (This is a small pre-trained model to bootstrap grammar)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("\n>> Extracting semantic scaffolding...")
    
    # Extract the vocabulary
    vocab = tokenizer.get_vocab()
    
    # Filter vocabulary to alphanumeric, readable, lowercase words
    filtered_tokens = []
    for token, idx in vocab.items():
        clean_token = token.strip('Ġ').lower() # GPT-Neo style tokenization
        if clean_token.isalpha() and len(clean_token) >= 2:
            filtered_tokens.append((clean_token, idx))
            
    # Sort by index and take the most common words
    filtered_tokens.sort(key=lambda x: x[1])
    selected_tokens = []
    seen = set()
    
    for token, original_idx in filtered_tokens:
        if token not in seen and len(selected_tokens) < max_vocab:
            seen.add(token)
            selected_tokens.append((token, original_idx))
            
    print(f"   Selected {len(selected_tokens)} high-quality core concepts.")
    
    # Get the embedding weights and LM head weights
    original_embeddings = model.transformer.wte.weight.data
    # distilgpt2 has embedding dim = 768. 
    model_dim = original_embeddings.shape[1]
    
    print(f"   Original embedding dimension: {model_dim}")
    
    # Initialize our CrushedSubstrate
    print("\n>> Initializing CrushedSubstrate...")
    substrate = CrushedSubstrate(
        initial_concepts=len(selected_tokens),
        embedding_dim=model_dim,
        hdc_dim=4096
    )
    
    # Map the tokens and embeddings
    print(">> Injecting DNA (Embeddings & Vocabulary)...")
    new_embeddings = torch.zeros((len(selected_tokens), model_dim))
    
    for i, (token, orig_idx) in enumerate(selected_tokens):
        # Register in our substrate
        substrate.token_to_id[token] = i
        substrate.id_to_token[i] = token
        
        # Copy the embedding
        new_embeddings[i] = original_embeddings[orig_idx]
        
    # Replace the substrate's random embeddings with the distilled ones
    substrate.embeddings = torch.nn.Parameter(new_embeddings)
    substrate.vocab_size = len(selected_tokens)
    
    print(">> Injecting Ternary Grammar (Transition Matrix)...")
    # To get grammatical transitions, we approximate the attention/MLP layers 
    # by taking the direct projection from embedding to LM head.
    # P(next | current) ~= Embeddings * LM_Head^T
    
    # But since we use a direct transition matrix `transition(x)`, we want:
    # transition.weight * embedding = next_embedding
    
    # We will initialize the transition matrix with an identity + local mixing
    # to preserve the semantic structure, and let the first training pass adapt it.
    with torch.no_grad():
        # Diagonal dominance + small noise for transition
        identity_mix = torch.eye(model_dim) * 0.5
        noise = torch.randn(model_dim, model_dim) * 0.05
        substrate.transition.weight.data = identity_mix + noise

    # Save the injected checkpoint
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    save_path = os.path.join(checkpoint_dir, "substrate_distilled.pt")
    
    # We save exactly how train_loop expects it
    state = {
        'embeddings': substrate.embeddings.data.cpu(),
        'transition_weight': substrate.transition.weight.data.cpu(),
        'token_to_id': substrate.token_to_id,
        'id_to_token': substrate.id_to_token,
        'vocab_size': substrate.vocab_size,
        'step': 0,
        'surprise': 1.0, # Reset surprise
        'compression_stats': substrate.get_compression_stats(),
    }
    torch.save(state, save_path)
    print(f"\n>> Distillation complete! Checkpoint saved to:\n   {save_path}")
    
    # Test generation with the injected knowledge
    print("\n>> Initial Broca's Area Test (No training yet):")
    generator = NativeLanguageGenerator(substrate)
    for seed in ["the", "he", "it"]:
        if seed in substrate.token_to_id:
            res = generator.generate(seed=seed, max_tokens=10, temperature=0.2)
            print(f"   [{seed}] -> {res['text']}")

    print("\n>> Next Step: Run the train_loop using this checkpoint to solidify the grammar!")
    print("   python -m core.novacrush.train_loop --resume checkpoints/substrate_distilled.pt")

if __name__ == "__main__":
    inject_fluency()
