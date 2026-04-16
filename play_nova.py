import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np

# Import all modules
from novamind.v10.tokenizer import Tokenizer
from novamind.v10.models.text_encoder import TextEncoder
from novamind.v10.models.jepa import JEPATrunk
from novamind.v10.models.rssm import RSSM
from novamind.v10.models.moe import SparseMoE
from novamind.v10.models.actor_critic import ActorCritic
from novamind.v10.models.symbolic import SymbolicHead
from novamind.v10.models.faiss_memory import FaissMemory
from novamind.v10.models.runtime import LocalRuntime

# 1. Ensure SPM model exists
def build_dummy_spm():
    # Build a small dummy sentencepiece model to reach vocab=16384
    # Sentencepiece requires sufficient characters to reach vocab sizes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'novamind', 'v10', 'tokenizer', 'spm_16k.model')
    
    if os.path.exists(model_path):
        return
        
    print("[INIT] Building initial BPE Tokenizer (16k size)...")
    text = "hello world agi " * 10000
    # Add many unique tokens
    for i in range(16500):
        text += f"word_{i} "
        
    with open('dummy.txt', 'w') as f:
        f.write(text)
        
    spm.SentencePieceTrainer.train(
        input='dummy.txt', 
        model_prefix=model_path.replace('.model', ''), 
        vocab_size=16384,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    os.remove('dummy.txt')

build_dummy_spm()

# 2. Instantiate Architecture
print("[INIT] Booting NovaMind v10 Component Graph...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer()
text_encoder = TextEncoder(vocab_size=16384, d_model=512).to(device)
jepa_trunk = JEPATrunk(embed_dim=512).to(device)
rssm = RSSM(action_dim=10, embed_dim=512).to(device)
moe = SparseMoE(d_model=512, num_experts=64, top_k=1).to(device)
actor_critic = ActorCritic(deter_dim=512, stoch_dim=32, stoch_classes=32, action_dim=10).to(device)
symbolic_head = SymbolicHead(embed_dim=512).to(device)
memory = FaissMemory(embed_dim=512)

# Initial hidden state
batch_size = 1
current_state = rssm.initial_state(batch_size, device=device)
last_action = torch.zeros(1, 10).to(device)

def tensor_cognitive_cycle(user_input):
    global current_state, last_action
    
    # 1. Encode text via BPE
    tokens = tokenizer.encode(user_input)
    if not tokens: # Empty input
        tokens = [tokenizer.unk_id]
        
    input_ids = torch.tensor([tokens]).to(device)
    
    # 2. Dense Embed -> JEPA Trunk
    embeddings = text_encoder(input_ids) # [1, seq, 512]
    jepa_out = jepa_trunk(embeddings) # [1, seq, 512]
    
    # Aggregate sequence (e.g. mean pool) to feed into RSSM step
    # Real representation relies on temporal dimension or aggregated context
    obs_embed = jepa_out.mean(dim=1) # [1, 512]
    
    # 3. RSSM World Model Step
    with torch.no_grad():
        prior, posterior, current_state = rssm.step(current_state, last_action, obs_embed)
        
        # 4. Sparse MoE Reasoning
        dense_latent = current_state['deter'] # [1, 512]
        # Expand for MoE [1, 1, 512]
        moe_out, _ = moe(dense_latent.unsqueeze(1)) 
        moe_out = moe_out.squeeze(1)
        
        # 5. Policy selects Action (EFE surrogate -> Actor Critic)
        # Flatten categorical stochastic state
        stoch_flat = current_state['stoch'].view(current_state['stoch'].size(0), -1)
        action, _ = actor_critic(moe_out, stoch_flat)
        last_action = action
        
        # 6. Optional Memory Read/Write
        # Example Write:
        memory.store(dense_latent.cpu().numpy())
        dist, ids, _ = memory.retrieve(dense_latent.cpu().numpy())
        
        # 7. Symbolic Generation
        programs, _ = symbolic_head.generate_program(moe_out, max_len=3)
        program_emitted = "".join(programs[0][:-1]) # Exclude EOS
        
        # Convert Action back to String
        # Since it's continuous space around 0, we can threshold to words for fun
        # Typically the "Emit Token" head would be a linear over vocab size here.
        simulated_logits = F.linear(action, torch.randn(16384, 10).to(device))
        sampled_id = simulated_logits.argmax(-1).item()
        word = tokenizer.decode([sampled_id])
        
    response = f"[RSSM-MoE] Think: {program_emitted} | [Emit]: {word} | [Mem-Idx]: {ids[0][0]}"
    return response

# Runtime patch
runtime = LocalRuntime(max_iterations=10)
runtime._cognitive_cycle = tensor_cognitive_cycle

print("\n--- NovaMind v10 ATIVADO --- (Digite 'exit' para sair)")
# To prevent infinite blocking in terminal, we provide a fixed script sequence if testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        outputs = runtime.start(iter(["Ola, mundo", "Quem e voce?", "Aja!"]))
        for o in outputs:
            print(o)
    else:
        runtime.start()
