import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ===== GARANTIR GPU (Restringir o processamento à GPU) =====
if not torch.cuda.is_available():
    raise SystemError("Erro Crítico: O processamento foi travado estritamente para a GPU, mas nenhuma GPU CUDA foi encontrada no ambiente!")

# Garantir a primeira GPU apenas (opcional caso haja múltiplas)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = 'cuda'
print(f"[INIT] Limitando o processamento para a GPU principal: {torch.cuda.get_device_name(0)}")

# ===== IMPORTAR COMPONENTES NOVA MIND =====
from novamind.v10.tokenizer import Tokenizer
from novamind.v10.models.text_encoder import TextEncoder
from novamind.v10.models.jepa import JEPATrunk
from novamind.v10.models.rssm import RSSM
from novamind.v10.models.moe import SparseMoE
from novamind.v10.models.actor_critic import ActorCritic
from novamind.v10.models.symbolic import SymbolicHead
from novamind.v10.models.faiss_memory import FaissMemory

# 1. BPE Tokenizer Fake (16k)
def build_dummy_spm():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'novamind', 'v10', 'tokenizer', 'spm_16k.model')
    if os.path.exists(model_path):
        return
    print("[INIT] Building initial BPE Tokenizer (16k size)...")
    text = "hello world agi " * 10000
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

# 2. Instantiate Architecture in GPU
print("[INIT] Booting NovaMind v10 Component Graph na GPU...")
tokenizer = Tokenizer()
text_encoder = TextEncoder(vocab_size=16384, d_model=512).to(device)
jepa_trunk = JEPATrunk(embed_dim=512).to(device)
rssm = RSSM(action_dim=10, embed_dim=512).to(device)
moe = SparseMoE(d_model=512, num_experts=64, top_k=1).to(device)
actor_critic = ActorCritic(deter_dim=512, stoch_dim=32, stoch_classes=32, action_dim=10).to(device)
symbolic_head = SymbolicHead(embed_dim=512).to(device)
memory = FaissMemory(embed_dim=512)

# Global states
batch_size = 1
current_state = rssm.initial_state(batch_size, device=device)
last_action = torch.zeros(1, 10).to(device)

def tensor_cognitive_cycle(user_input):
    global current_state, last_action
    
    tokens = tokenizer.encode(user_input)
    if not tokens:
        tokens = [tokenizer.unk_id]
        
    input_ids = torch.tensor([tokens]).to(device)
    
    # Forward passes (apenas GPU)
    embeddings = text_encoder(input_ids)
    jepa_out = jepa_trunk(embeddings)
    obs_embed = jepa_out.mean(dim=1)
    
    with torch.no_grad():
        prior, posterior, current_state = rssm.step(current_state, last_action, obs_embed)
        
        dense_latent = current_state['deter']
        moe_out, _ = moe(dense_latent.unsqueeze(1)) 
        moe_out = moe_out.squeeze(1)
        
        stoch_flat = current_state['stoch'].view(current_state['stoch'].size(0), -1)
        action, _ = actor_critic(moe_out, stoch_flat)
        last_action = action
        
        # CPU call to faiss
        memory.store(dense_latent.cpu().numpy())
        dist, ids, _ = memory.retrieve(dense_latent.cpu().numpy())
        
        programs, _ = symbolic_head.generate_program(moe_out, max_len=3)
        program_emitted = "".join(programs[0][:-1])
        
        simulated_logits = F.linear(action, torch.randn(16384, 10).to(device))
        sampled_id = simulated_logits.argmax(-1).item()
        word = tokenizer.decode([sampled_id])
        
    # Retornamos dict estruturado para API
    return {
        "thought": program_emitted,
        "emit": word,
        "memIdx": float(ids[0][0])
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NovaMind AGI - Localhost</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --container-bg: rgba(255, 255, 255, 0.03);
            --text-color: #e2e8f0;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --blur: blur(12px);
        }
        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            background-image: radial-gradient(circle at 15% 50%, rgba(59, 130, 246, 0.15), transparent 25%),
                              radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.15), transparent 25%);
            color: var(--text-color);
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            overflow: hidden;
        }
        .chat-container {
            width: 100%;
            max-width: 600px;
            height: 80vh;
            background: var(--container-bg);
            backdrop-filter: var(--blur);
            -webkit-backdrop-filter: var(--blur);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .header {
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            font-weight: 600;
            letter-spacing: 1px;
            color: var(--accent);
            text-transform: uppercase;
        }
        .messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .msg {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 0.95rem;
            line-height: 1.4;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .msg.user {
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.3);
            align-self: flex-end;
            border-bottom-right-radius: 0;
        }
        .msg.bot {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            align-self: flex-start;
            border-bottom-left-radius: 0;
            display: flex;
            flex-direction: column;
        }
        .bot .thought {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-bottom: 5px;
            font-style: italic;
        }
        .bot .emit {
            color: #e2e8f0;
        }
        .input-area {
            padding: 20px;
            border-top: 1px solid rgba(255,255,255,0.05);
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            background: rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.1);
            padding: 12px 15px;
            border-radius: 10px;
            color: white;
            font-family: inherit;
            outline: none;
            transition: border 0.3s;
        }
        input:focus {
            border-color: var(--accent);
        }
        button {
            background: var(--accent);
            color: white;
            border: none;
            padding: 0 20px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.3s;
        }
        button:hover {
            background: var(--accent-hover);
        }
        .loading {
            font-size: 0.8rem;
            color: #94a3b8;
            align-self: flex-start;
            margin-left: 20px;
            display: none;
        }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="header">🟢 NovaMind AGI (GPU Local)</div>
    <div class="messages" id="messages">
        <div class="msg bot"><span class="emit">GPU limit configurado. Servidor Localhost Ativado. Pode me enviar uma mensagem.</span></div>
    </div>
    <div class="loading" id="loading">NovaMind pensando...</div>
    <div class="input-area">
        <input type="text" id="userInput" placeholder="Envie algo para o AGI..." onkeypress="if(event.key === 'Enter') sendMessage()">
        <button onclick="sendMessage()">Enviar</button>
    </div>
</div>

<script>
    const msgs = document.getElementById('messages');
    const loading = document.getElementById('loading');
    
    async function sendMessage() {
        const input = document.getElementById('userInput');
        const text = input.value.trim();
        if(!text) return;
        
        msgs.innerHTML += `<div class="msg user">${text}</div>`;
        input.value = '';
        msgs.scrollTop = msgs.scrollHeight;
        
        loading.style.display = 'block';
        
        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            
            loading.style.display = 'none';
            msgs.innerHTML += `
                <div class="msg bot">
                    <span class="thought">[RSSM-MoE] Think: ${data.thought} (Mem: ${data.memIdx})</span>
                    <span class="emit">${data.emit}</span>
                </div>
            `;
            msgs.scrollTop = msgs.scrollHeight;
        } catch(e) {
            loading.style.display = 'none';
            msgs.innerHTML += `<div class="msg bot" style="color:#ef4444;">Erro de conexão com AGI.</div>`;
        }
    }
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    text = data.get("text", "")
    try:
        response_dict = tensor_cognitive_cycle(text)
        return jsonify(response_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n--- NovaMind SERVIDOR LOCALHOST INICIADO NA GPU ---")
    app.run(host="127.0.0.1", port=5000, debug=False)
