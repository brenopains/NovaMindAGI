"""
NovaCrush -- Continuous Training Loop
========================================
Feed text data continuously into the CrushedSubstrate.
The more it trains, the better it speaks.

Usage:
    python -m core.novacrush.train_loop
    
    # Or with custom data directory:
    python -m core.novacrush.train_loop --data path/to/text/files

The trainer:
    1. Loads all .txt files from the data directory
    2. Splits them into sentences/phrases
    3. Feeds them continuously to the substrate
    4. Every N steps, generates sample output to show progress
    5. Saves checkpoints periodically
    
You can also paste text directly into the training data folder
while the trainer is running -- it will pick up new files automatically.
"""

import os
import sys
import re
import time
import json
import glob
import torch
import numpy as np
import argparse
from typing import List, Dict, Optional

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.novacrush.crushed_substrate import CrushedSubstrate
from core.novacrush.language_gen import NativeLanguageGenerator


def load_text_files(data_dir: str) -> List[str]:
    """Load all text files from directory."""
    texts = []
    patterns = ['*.txt', '*.md', '*.py', '*.json', '*.csv', '*.html']
    
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(data_dir, '**', pattern), recursive=True):
            # Skip venv, __pycache__, .git
            if any(skip in filepath for skip in ['venv', '__pycache__', '.git', 'node_modules']):
                continue
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        texts.append(content)
            except Exception:
                continue
    
    return texts


def extract_sentences(texts: List[str]) -> List[List[str]]:
    """Split texts into tokenized sentences."""
    sentences = []
    
    for text in texts:
        # Split on sentence boundaries
        raw_sentences = re.split(r'[.!?\n]+', text)
        
        for sent in raw_sentences:
            # Tokenize: keep only ASCII alphanumeric words
            words = [w.strip().lower() for w in re.split(r'\W+', sent) if w.strip()]
            # Filter non-ASCII tokens (avoid Windows encoding crash)
            words = [w for w in words if w.isascii()]
            
            # Filter: sentences of reasonable length
            if 3 <= len(words) <= 50:
                sentences.append(words)
    
    return sentences


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def safe_print(text: str):
    """Print text safely on Windows (strip non-ASCII)."""
    clean = text.encode('ascii', errors='replace').decode('ascii')
    print(clean)


def run_training(data_dir: str = None, 
                 embedding_dim: int = 32,
                 initial_concepts: int = 64,
                 hdc_dim: int = 4096,
                 epochs: int = 100,
                 sample_interval: int = 50,
                 checkpoint_interval: int = 200,
                 checkpoint_dir: str = None):
    """
    Run continuous training loop.
    
    This is the core training function. It:
    1. Creates or loads a CrushedSubstrate
    2. Loads training text
    3. Trains continuously
    4. Generates sample output periodically
    5. Saves checkpoints
    """
    
    # Default data directory: the project itself (learn from own source code + docs)
    if data_dir is None:
        data_dir = PROJECT_ROOT
    
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("=" * 70)
    print("  NovaCrush Continuous Training Loop")
    print("=" * 70)
    print(f"  Data directory: {data_dir}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
    
    # Load training data
    print("\n  Loading text data...")
    texts = load_text_files(data_dir)
    sentences = extract_sentences(texts)
    np.random.shuffle(sentences)
    
    print(f"  Files loaded: {len(texts)}")
    print(f"  Sentences extracted: {len(sentences)}")
    total_tokens = sum(len(s) for s in sentences)
    print(f"  Total tokens: {total_tokens:,}")
    
    if len(sentences) == 0:
        print("\n  ERROR: No training data found!")
        print(f"  Put .txt files in: {data_dir}")
        return
    
    # Create substrate
    print("\n  Creating CrushedSubstrate...")
    substrate = CrushedSubstrate(
        initial_concepts=initial_concepts,
        embedding_dim=embedding_dim,
        hdc_dim=hdc_dim
    )
    generator = NativeLanguageGenerator(substrate)
    
    # Training loop
    print(f"\n  Training for {epochs} epochs ({epochs * len(sentences)} steps)...")
    print(f"  Sample generation every {sample_interval} steps")
    print(f"  Checkpoint every {checkpoint_interval} steps")
    print("-" * 70)
    
    global_step = 0
    t_start = time.time()
    surprise_window = []
    best_surprise = float('inf')
    
    for epoch in range(epochs):
        # Shuffle each epoch
        np.random.shuffle(sentences)
        epoch_surprise = 0.0
        epoch_steps = 0
        
        for i, sentence in enumerate(sentences):
            # Train on this sentence
            surprise = substrate.continuous_train(sentence)
            
            global_step += 1
            epoch_steps += 1
            epoch_surprise += surprise
            surprise_window.append(surprise)
            if len(surprise_window) > 100:
                surprise_window.pop(0)
            
            # Progress display
            if global_step % sample_interval == 0:
                avg_surprise = np.mean(surprise_window)
                elapsed = time.time() - t_start
                steps_per_sec = global_step / max(0.001, elapsed)
                
                # Track improvement
                if avg_surprise < best_surprise:
                    best_surprise = avg_surprise
                    improved = " *BEST*"
                else:
                    improved = ""
                
                print(f"\n  [Step {global_step:,} | Epoch {epoch+1}/{epochs} | "
                      f"{format_time(elapsed)} | {steps_per_sec:.1f} steps/s]")
                print(f"  Surprise: {avg_surprise:.4f}{improved}")
                print(f"  Vocab: {substrate.vocab_size} tokens")
                print(f"  Sparsity: {substrate.transition.get_compression_stats()['sparsity']:.1%}")
                
                # Generate samples
                print(f"\n  --- Generated Samples (T=0.3) ---")
                for seed_word in ['the', 'learning', 'neural']:
                    if seed_word in substrate.token_to_id:
                        result = generator.generate(
                            seed=seed_word, max_tokens=12, temperature=0.3
                        )
                        safe_print(f"    \"{result['text']}\"")
                
                # Random seed sample
                result = generator.generate(max_tokens=15, temperature=0.5)
                safe_print(f"    (random) \"{result['text']}\"")
                
                print(f"  ---------------------------------")
            
            # Checkpoint
            if global_step % checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    checkpoint_dir, 
                    f"substrate_step{global_step}.pt"
                )
                save_checkpoint(substrate, ckpt_path, global_step, avg_surprise)
                print(f"  >> Checkpoint saved: {ckpt_path}")
        
        # End of epoch summary
        avg_epoch = epoch_surprise / max(1, epoch_steps)
        print(f"\n  === Epoch {epoch+1} complete | "
              f"Avg surprise: {avg_epoch:.4f} | "
              f"Vocab: {substrate.vocab_size} ===")
    
    # Final summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {format_time(elapsed)}")
    print(f"  Total steps: {global_step:,}")
    print(f"  Final vocab: {substrate.vocab_size}")
    print(f"  Best surprise: {best_surprise:.4f}")
    print("=" * 70)
    
    # Final generation showcase
    print("\n  === Final Generation Showcase ===")
    for temp in [0.1, 0.3, 0.5, 0.8]:
        result = generator.generate(max_tokens=20, temperature=temp)
        safe_print(f"  T={temp}: \"{result['text']}\"")
    
    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "substrate_final.pt")
    save_checkpoint(substrate, final_path, global_step, best_surprise)
    print(f"\n  Final checkpoint: {final_path}")
    
    return substrate, generator


def save_checkpoint(substrate: CrushedSubstrate, path: str, 
                    step: int, surprise: float):
    """Save substrate state to disk."""
    state = {
        'embeddings': substrate.embeddings.data.cpu(),
        'transition_weight': substrate.transition.weight.data.cpu(),
        'token_to_id': substrate.token_to_id,
        'id_to_token': substrate.id_to_token,
        'vocab_size': substrate.vocab_size,
        'step': step,
        'surprise': surprise,
        'compression_stats': substrate.get_compression_stats(),
    }
    torch.save(state, path)


def load_checkpoint(path: str, embedding_dim: int = 32) -> CrushedSubstrate:
    """Load substrate from checkpoint."""
    state = torch.load(path, map_location='cpu')
    
    substrate = CrushedSubstrate(
        initial_concepts=state['vocab_size'],
        embedding_dim=embedding_dim,
    )
    
    with torch.no_grad():
        substrate.embeddings = torch.nn.Parameter(state['embeddings'])
        substrate.transition.weight = torch.nn.Parameter(state['transition_weight'])
    
    substrate.token_to_id = state['token_to_id']
    substrate.id_to_token = state['id_to_token']
    substrate.vocab_size = state['vocab_size']
    
    return substrate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NovaCrush Continuous Training')
    parser.add_argument('--data', type=str, default=None,
                       help='Directory with training text files')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--dim', type=int, default=32,
                       help='Embedding dimension')
    parser.add_argument('--concepts', type=int, default=64,
                       help='Initial concept capacity')
    parser.add_argument('--sample-interval', type=int, default=100,
                       help='Steps between sample generations')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                       help='Steps between checkpoints')
    
    args = parser.parse_args()
    
    run_training(
        data_dir=args.data,
        embedding_dim=args.dim,
        initial_concepts=args.concepts,
        epochs=args.epochs,
        sample_interval=args.sample_interval,
        checkpoint_interval=args.checkpoint_interval,
    )
