import os
import sys
import asyncio
import json
import base64
import io
import random
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datasets import load_dataset

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from novamind.v10.tokenizer import Tokenizer
from novamind.v10.models.actor_critic import ActorCritic
from novamind.v10.models.symbolic import SymbolicHead
from novamind.v10.models.faiss_memory import FaissMemory
from novamind.v10.models.text_encoder import TextEncoder
from novamind.v10.models.jepa import JEPATrunk
from novamind.v10.models.rssm import RSSM
from novamind.v10.models.moe import SparseMoE
from novamind.v10.models.neuromorphic import LIFNeuron
from novamind.v10.models.vision_encoder import VisionVQVAE
from novamind.v10.models.audio_encoder import AudioEncoder
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(autonomous_mind_loop())
    yield
    task.cancel()

app = FastAPI(title="NovaMind v10 AGI Multimodal", lifespan=lifespan)
ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
app.mount("/static", StaticFiles(directory=ui_dir), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(ui_dir, "index.html"))

print("[Backend] Booting NovaMind AGI Multimodal Cortex (A40 Massive Mode)...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Massive run constraints
D_MODEL = 2048
STOCH_DIM = 256
STOCH_CLASSES = 256

tokenizer = Tokenizer()
text_encoder = TextEncoder(vocab_size=16384, d_model=D_MODEL).to(device)

# --- SENSORY ENCODERS ---
vision_encoder_core = VisionVQVAE(num_embeddings=512, embedding_dim=64).to(device)
# Map 512 discrete vision codes into standard D_MODEL sequence elements
vision_projection = torch.nn.Embedding(512, D_MODEL).to(device)

audio_encoder_core = AudioEncoder(in_channels=1, d_model=D_MODEL).to(device)

jepa_trunk = JEPATrunk(embed_dim=D_MODEL).to(device)
rssm = RSSM(action_dim=10, embed_dim=D_MODEL, deter_dim=D_MODEL, stoch_dim=STOCH_DIM, stoch_classes=STOCH_CLASSES, hidden_dim=D_MODEL).to(device)
moe = SparseMoE(d_model=D_MODEL, num_experts=8, top_k=2).to(device) # expanded experts for A40
actor_critic = ActorCritic(deter_dim=D_MODEL, stoch_dim=STOCH_DIM, stoch_classes=STOCH_CLASSES, action_dim=10).to(device)
symbolic_head = SymbolicHead(embed_dim=D_MODEL).to(device)
memory = FaissMemory(embed_dim=D_MODEL)
snn_adapter = LIFNeuron(action_dim=10, threshold=0.75, leak_decay=0.85).to(device)

emit_head = torch.nn.Linear(10, 16384).to(device)

all_params = (
    list(text_encoder.parameters()) + list(jepa_trunk.parameters()) +
    list(rssm.parameters()) + list(moe.parameters()) +
    list(actor_critic.parameters()) + list(symbolic_head.parameters()) +
    list(emit_head.parameters()) + list(vision_encoder_core.parameters()) +
    list(vision_projection.parameters()) + list(audio_encoder_core.parameters())
)
optimizer = torch.optim.AdamW(all_params, lr=3e-4)

batch_size = 1
current_state = rssm.initial_state(batch_size, device=device)
last_action = torch.zeros(1, 10).to(device)

def tensor_step(input_text: str, input_image: torch.Tensor = None, input_audio: torch.Tensor = None, is_autonomous: bool = False, target_text: str = None) -> dict:
    global current_state, last_action
    
    # 1. TEXT STREAM
    tokens = tokenizer.encode(input_text)
    if not tokens: tokens = [tokenizer.unk_id]
    input_ids = torch.tensor([tokens]).to(device)
    embeddings = text_encoder(input_ids) # [1, seq, D_MODEL]
    
    sequence_tensors = [embeddings]
    
    # 2. VISION STREAM
    if input_image is not None:
        # Expected image: [B, 3, 64, 64]
        input_image = input_image.to(device)
        quantized_indices, _ = vision_encoder_core(input_image) # [B, 64]
        vis_emb = vision_projection(quantized_indices) # [B, 64, D_MODEL]
        sequence_tensors.append(vis_emb)
        
    # 3. AUDIO STREAM
    if input_audio is not None:
        # Expected audio: [B, 1, T] 
        input_audio = input_audio.to(device)
        aud_emb = audio_encoder_core(input_audio) # [B, seq_a, D_MODEL]
        sequence_tensors.append(aud_emb)

    # UNIFIED MULTIMODAL CONCATENATION ALONG TIME (SEQUENCE)
    fused_embeddings = torch.cat(sequence_tensors, dim=1)
    
    # FORWARD PASS INTO CORTEX
    jepa_out = jepa_trunk(fused_embeddings)
    obs_embed = jepa_out.mean(dim=1)
    
    prior, posterior, current_state = rssm.step(current_state, last_action, obs_embed)
    dense_latent = current_state['deter']
    
    # MOE Expert routing requires [B, SeqLength, D_MODEL]
    # In earlier versions it was [B, 1, D_MODEL].
    moe_out, _ = moe(dense_latent.unsqueeze(1))
    moe_out = moe_out.squeeze(1)
    
    stoch_flat = current_state['stoch'].view(current_state['stoch'].size(0), -1)
    continuous_action, value = actor_critic(moe_out, stoch_flat)
    
    spikes, membrane_potentials = snn_adapter(continuous_action.abs())
    action = continuous_action
    
    # Latent Memory Renderer
    latent_geom = current_state['deter'][0].detach().cpu().numpy()
    latent_geom = ((latent_geom - latent_geom.min()) / (latent_geom.max() - latent_geom.min() + 1e-5) * 255).astype('uint8')
    im = Image.fromarray(latent_geom.reshape(32, 64), mode='L').convert("P")
    im.putpalette([int(x) for x in range(256)] * 3)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64_latent_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    loss_val = 0.0
    emitted_text = ""
    emitted_modality = "text"
    emitted_media_b64 = None
    
    # 1. Modality Decision (Free Will Selection Disabled per user request)
    emitted_modality = "text"
    
    if is_autonomous:
        # Fast Training Mode (Single Token)
        gen_logits = emit_head(action) # [1, 16384]
        sampled_id = gen_logits.argmax(-1).item()
        emitted_text = tokenizer.decode([sampled_id])
        
        if target_text is not None:
            target_tokens = tokenizer.encode(target_text)
            if not target_tokens: target_tokens = [tokenizer.unk_id]
            target_ids = torch.tensor(target_tokens[-1:]).to(device)
            loss = F.cross_entropy(gen_logits, target_ids)
            loss.backward()
            loss_val = float(loss.item())

        last_action = action.detach()
        current_state = {k: v.detach() for k, v in current_state.items()}
    else:
        # Interactive Mode (Autoregressive Generation & Media)
        if emitted_modality == "text":
            gen_words = []
            loop_state = {k: v.detach() for k, v in current_state.items()}
            loop_action = action.detach()
            
            for _ in range(8): # Generate 8 token sentence
                gen_logits = emit_head(loop_action)
                sampled_id = gen_logits.argmax(-1).item()
                word = tokenizer.decode([sampled_id])
                gen_words.append(word)
                
                # Feedback loop
                next_input_ids = torch.tensor([[sampled_id]]).to(device)
                next_emb = text_encoder(next_input_ids)
                next_prior, next_posterior, loop_state = rssm.step(loop_state, loop_action, next_emb.squeeze(1))
                
                next_moe, _ = moe(loop_state['deter'].unsqueeze(1))
                stoch_flat = loop_state['stoch'].view(loop_state['stoch'].size(0), -1)
                next_action, _ = actor_critic(next_moe.squeeze(1), stoch_flat)
                loop_action = next_action.detach()
                
            emitted_text = " ".join(gen_words)
            
        last_action = action.detach()
        current_state = {k: v.detach() for k, v in current_state.items()}
    
    programs, _ = symbolic_head.generate_program(moe_out.detach(), max_len=3)
    program_emitted = "".join(programs[0][:-1])
    ast_nodes = [{"id": f"n{i}", "label": p} for i, p in enumerate(programs[0][:-1])]
    ast_edges = [{"from": f"n{i}", "to": f"n{i+1}"} for i in range(len(ast_nodes)-1)]
    
    return {
        "is_autonomous": is_autonomous,
        "input_tokens": tokens,
        "rssm_prior_mean": float(prior['logits'].mean().item()),
        "moe_expert_idx": 0, 
        "symbolic_prog": program_emitted,
        "symbolic_ast": {"nodes": ast_nodes, "edges": ast_edges},
        "minds_eye_b64": b64_latent_img,
        "mem_index": 0,
        "action_vector": [float(x) for x in action[0][:5].detach().cpu().numpy()], 
        "spikes": [int(x) for x in spikes[0][:5].cpu().numpy()],
        "membrane_potentials": [float(x) for x in membrane_potentials[0][:5].detach().cpu().numpy()],
        "emitted_word": emitted_text,
        "emitted_modality": emitted_modality,
        "emitted_media_b64": emitted_media_b64,
        "loss_val": loss_val
    }

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.autonomous_loop_running = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

async def autonomous_mind_loop():
    manager.autonomous_loop_running = True
    print("[MIND] Initializing MASSIVE Multimodal HuggingFace Streams (FineWeb / MNIST / LibriSpeech)...")
    
    try:
        # Puxando o FineWeb (O mesmo que treinou o Llama 3) para Inteligência de texto real
        ds_text = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        ds_vision = load_dataset("mnist", split="train", streaming=True)
        ds_audio = load_dataset("openslr/librispeech_asr", "default", split="validation", revision="refs/convert/parquet", streaming=True)
        
        text_iter = iter(ds_text)
        vision_iter = iter(ds_vision)
        audio_iter = iter(ds_audio)
    except Exception as e:
        print(f"[MIND] Stream boot error: {e}. Ensure libraries are installed.")
        return

    trans_vis = T.Compose([T.Resize((64, 64)), T.Grayscale(num_output_channels=3), T.ToTensor()])
    
    global_step = 0
    best_loss = 999.0
    
    while True:
        await asyncio.sleep(0.01) # Yield to event loop slightly
        if not manager.active_connections:
            await asyncio.sleep(1.0)
            continue
            
        try:
            # 1. Fetch Multimodal Elements
            text_sample = next(text_iter)['text'].strip()
            if not text_sample: continue
                
            img_sample = next(vision_iter)['image']
            img_tensor = trans_vis(img_sample).unsqueeze(0) # [1, 3, 64, 64]
            
            try:
                audio_sample = next(audio_iter)['audio']['array']
                audio_cropped = torch.tensor(audio_sample[:2000]).float().unsqueeze(0).unsqueeze(0)
            except Exception:
                # O codec C++ do PyTorch falhou. Alimentamos o cortex auditivo dela com "Ruído Branco" (White Noise) 
                # para não interromper o treinamento do Fineweb!
                audio_cropped = torch.randn((1, 1, 2000)).float()
                
        except StopIteration:
            text_iter = iter(ds_text); vision_iter = iter(ds_vision); audio_iter = iter(ds_audio)
            continue
        except Exception as e:
            print(f"[MIND EXCEPTION]: {e}")
            await asyncio.sleep(1.0)
            continue
            
        words = text_sample.split()
        if len(words) < 3: continue
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for i in range(len(words)-2):
            context = " ".join(words[i:i+2])
            target = words[i+2]
            
            # Pass everything. The model learns to ground words with the visual/auditory context!
            brain_state = tensor_step(input_text=context, input_image=img_tensor, input_audio=audio_cropped, is_autonomous=True, target_text=target)
            accumulated_loss += brain_state["loss_val"]
            
            payload = json.dumps({"type": "brain_cycle", "data": brain_state})
            for conn in manager.active_connections.copy():
                try:
                    await conn.send_text(payload)
                except WebSocketDisconnect:
                    manager.disconnect(conn)
            
            await asyncio.sleep(0.05)
            
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        global_step += 1
        
        avg_loss = accumulated_loss / (len(words)-2)
        if avg_loss > 0 and avg_loss < best_loss:
            best_loss = avg_loss
            if global_step % 20 == 0:
                torch.save({
                    'text_enc': text_encoder.state_dict(),
                    'vis_enc': vision_encoder_core.state_dict(),
                    'aud_enc': audio_encoder_core.state_dict(),
                    'jepa': jepa_trunk.state_dict(),
                    'rssm': rssm.state_dict(),
                    'actor': actor_critic.state_dict()
                }, "best_a40_multimodal.pt")
        
        await asyncio.sleep(0.1)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data_str = await websocket.receive_text()
            try:
                payload = json.loads(data_str)
                text_input = payload.get("text", "")
                media_b64 = payload.get("media")
                media_type = payload.get("media_type", "")
            except:
                text_input = data_str
                media_b64 = None
                media_type = ""
                
            input_img_tensor = None
            input_aud_tensor = None
            
            if media_b64:
                try:
                    if "base64," in media_b64:
                        _, encoded = media_b64.split("base64,", 1)
                    else:
                        encoded = media_b64
                    
                    raw_bytes = base64.b64decode(encoded)
                    
                    if "image" in media_type:
                        im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                        trans_vis = T.Compose([T.Resize((64, 64)), T.ToTensor()])
                        input_img_tensor = trans_vis(im).unsqueeze(0).to(device)
                    elif "audio" in media_type or "video" in media_type:
                        # Attempt torchaudio for microphone
                        import torchaudio
                        # torchaudio can often process from BytesIO if backend supports it. Better to write to temp file.
                        tmp_path = "tmp_audio.webm"
                        with open(tmp_path, "wb") as f:
                            f.write(raw_bytes)
                        try:
                            waveform, _ = torchaudio.load(tmp_path)
                            if waveform.shape[0] > 1:
                                waveform = waveform.mean(dim=0, keepdim=True)
                            audio_cropped = waveform[:, :2000]
                            # pad if too short
                            if audio_cropped.shape[1] < 2000:
                                audio_cropped = F.pad(audio_cropped, (0, 2000 - audio_cropped.shape[1]))
                            input_aud_tensor = audio_cropped.unsqueeze(0).to(device)
                        finally:
                            if os.path.exists(tmp_path): os.remove(tmp_path)
                except Exception as e:
                    print(f"Media Decode Error: {e}")

            thinking_msg = json.dumps({"type": "status", "message": "processing", "input": text_input})
            await websocket.send_text(thinking_msg)
            
            await asyncio.sleep(0.1)
            brain_state = tensor_step(
                input_text=text_input, 
                input_image=input_img_tensor, 
                input_audio=input_aud_tensor, 
                is_autonomous=False
            )
            
            payload = json.dumps({"type": "brain_cycle", "data": brain_state})
            await websocket.send_text(payload)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
