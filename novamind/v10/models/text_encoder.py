import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 16384, d_model: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Eliminates the old 80,000 fixed embedding and introduces a precise vocab-mapped embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0  # Tokenizer pad_id is 0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps a sequence of token IDs to dense vectors.
        
        Args:
            x: Tensor of shape [batch_size, seq_len] containing token ints.
            
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        return self.embedding(x)
