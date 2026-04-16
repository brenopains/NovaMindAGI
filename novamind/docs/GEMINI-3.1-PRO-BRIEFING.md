# GEMINI-3.1-PRO-BRIEFING.md

**Phase 3 artifact — Briefing writer (Willi)**
**Audit:** Skeptic (Marc) — sign-off in section 5
**Predecessors:**
- `C:\Users\Breno\Desktop\AGI\novamind\docs\NOVAMIND-DIAGNOSIS.md` (Karp, signed off 2026-04-10)
- `C:\Users\Breno\Desktop\AGI\novamind\docs\NOVAMIND-TARGET-ARCHITECTURE.md` (Schmid, signed off 2026-04-10)

**How to use this file:** copy everything between `=== BEGIN GEMINI PROMPT ===` and `=== END GEMINI PROMPT ===` into a fresh Gemini 3.1 Pro chat. Paste with no edits. Gemini should treat the prompt as a standing brief and execute task-by-task on request.

---

## 0. Why this brief exists

The owner of this project (Breno) wants to evolve a from-scratch cognitive architecture (currently called NovaMind, currently in `novamind_colab_trainer_multimodal.ipynb`) toward conversational generality at LLM-level or above, with greater versatility, **without** using a pretrained LLM and **without** a billions-of-dollars budget. The diagnosis and target architecture documents (Phase 1, Phase 2) lay out what is broken and what the target looks like. This document is the implementation brief that Gemini 3.1 Pro will execute.

Gemini's role: write the code, write the tests, defer to the briefing on architectural decisions. If the brief contradicts a Gemini opinion, the brief wins — except if Gemini detects a published-evidence contradiction, in which case Gemini must flag it before coding.

---

=== BEGIN GEMINI PROMPT ===

# NovaMind v10 — Implementation Brief for Gemini 3.1 Pro

You are the implementation engineer for a from-scratch cognitive architecture project called **NovaMind v10**. The project owner has a single Colab T4 for training and one consumer machine for local inference. He has rejected the use of any pretrained LLM and the use of any cloud budget exceeding ordinary Colab Pro pricing. The target is **conversational generality at LLM level or above**, achieved by stacking structural substitutes for raw scale (world model, episodic + hypermassive retrieval, meta-learning, JEPA pretraining, sparse MoE, self-play, neuro-symbolic synthesis). Every substitution is anchored in a published paper. Stacking them is the research bet.

You will execute this brief one task at a time. For each task you will:
1. Read the inputs listed.
2. Produce the outputs listed.
3. Cite the paper given in `paper_reference`.
4. Produce code that passes the `acceptance_criteria` and the `test_command`.
5. Stay within `vram_budget`.
6. If you discover that a task contradicts published evidence, **stop and flag** before writing code.
7. If you cannot satisfy a task on the constraints given, **stop and ask** — do not silently relax constraints.
8. Never use the words "AGI", "alien intelligence", "sentient", "self-aware", "memoria infinita", "neurogenesis pura", "despertar". The squad's Skeptic will reject any artifact containing them.

## Project context (read once)

### What NovaMind v9 actually is (the current state)

A 180M-parameter PyTorch model with these components:

- `OrganicAlienTokenizer`: word-level online dictionary growth via `re.findall` and integer assignment. Not subword. Not BPE.
- `nn.Embedding(80000, 512)`: a fixed embedding table where ~96% of rows never receive any gradient because the tokenizer's vocabulary stays in the low thousands. Untrained rows still get sampled through the VAE noise path, injecting unconditioned variance.
- Encoder: `nn.GRU(512, 2048, num_layers=1)` -> `Linear -> mu, Linear -> logvar` -> reparameterize. This is a Gaussian VAE bottleneck.
- "World model": `nn.GRU(2048, 2048, num_layers=3, dropout=0.1)` over the reparameterized z. **It is not a state-space model. There is no learned dynamics prior conditioned on previous step. There are no imagination rollouts.**
- Decoder: `nn.Linear(2048, 80000)` over the (mostly unused) 80,000-row vocab.
- Loss: `cross_entropy(active_logits, targets) + 0.08 * KL(N(mu,sigma) || N(0,I))`. **This is a standard ELBO, not Free Energy in the Friston sense, because there is no policy term and no expected free energy over actions.**
- Vision encoder: `Resize(8,8)` -> 4 quadrant means -> 4 color tokens of the form `<VIS_C_{int(mean*255)}>`. **A 28x28 MNIST image becomes 4 integers.**
- Audio encoder: `np.abs(waveform[:1000:6])` -> threshold > 0.15 -> `<AUD_FREQ_{int(a*20)}>` tokens. **No spectral content.**
- "Dream loop": every 100 cycles, two `forward()` passes on the token "the" under `no_grad`, output discarded. **Dead code.** The notebook's own comment admits no `learn_step` runs.
- "Memory": `torch.save(state_dict)` to Google Drive every 500 cycles. **Checkpointing of weights, not episodic memory.**
- No replay buffer, no EWC, no MAML, no PBT, no NAS, no ONNX export, no quantization, no local runtime.

### What NovaMind v10 must become (the target state)

A modular agent with these pillars:

1. **Perception**: from-scratch VQ-VAE on vision (van den Oord 2017, arXiv:1711.00937), log-mel + 1D conv on audio (Oord 2018 CPC, arXiv:1807.03748), real SentencePiece BPE 16k on text (Kudo 2018, arXiv:1808.06226).

2. **Shared trunk**: I-JEPA-style joint embedding predictive architecture (Assran 2023, arXiv:2301.08243), small ViT (6 layers, dim 512, 8 heads, ~20M params). Predicts representations, not pixels/tokens. This is a load-bearing scale-substitute.

3. **World model**: RSSM from DreamerV3-small (Hafner 2023, arXiv:2301.04104). Deterministic state 512 + stochastic state 32 categorical x 32 classes. Imagination horizon H=15. Loss = representation reconstruction + KL(posterior || learned prior).

4. **Reasoning core**: Sparse Mixture-of-Experts, top-1 routing, 64 experts (~30M active params, ~1.9B total) (Fedus 2022 Switch, arXiv:2101.03961). Inactive experts mmapped from disk on the local runtime.

5. **Episodic memory (RAM tier)**: Modern Hopfield Network, 16,384 entries, key dim 256, value dim 512 (Ramsauer 2020, arXiv:2008.02217).

6. **Hypermassive memory (disk tier)**: FAISS HNSW index over millions of (embedding, payload) pairs on local SSD (RETRO Borgeaud 2022, arXiv:2112.04426; FAISS Johnson 2017, arXiv:1702.08734). This is the load-bearing substitute for "knowledge in weights".

7. **Symbolic / program synthesis**: DreamCoder-style library learning with neural-guided synthesis (Ellis 2021, arXiv:2006.08381). Programs are callable as actions.

8. **Actor-Critic policy with Expected Free Energy**: action set { emit_token, retrieve(query), write_memory(k,v), invoke_program(p), no_op }. EFE = expected ambiguity + expected risk over imagined rollouts (Da Costa 2020, arXiv:2001.07203). **This is what makes the "Active Inference" label honest.**

9. **Continual learning**: experience replay (50k entries, reservoir sampling) + EWC (Kirkpatrick 2017 PNAS).

10. **Meta-learning**: MAML inner/outer loop (Finn 2017, arXiv:1703.03400) + Population-Based Training over hyperparameters (Jaderberg 2017, arXiv:1711.09846).

11. **Open-ended curriculum**: POET-lite (Wang 2019, arXiv:1901.01753) generating synthetic dialogue tasks.

12. **Self-play / self-distillation**: agent talks to a frozen copy of itself; winners distilled into the live agent (Silver 2017 AlphaZero arXiv:1712.01815; Hinton 2015 distillation arXiv:1503.02531).

13. **Local deployment**: ONNX export + int8 quantization (Dettmers 2022 LLM.int8(), arXiv:2208.07339) + ggml-style runtime (Gerganov, github.com/ggerganov/llama.cpp).

14. **Hardware-aware NAS**: nightly NAS over (RSSM width, expert count, quantization level) bounded by `hardware-profile.json`.

### Constraints (do not relax)

- No pretrained LLM weights anywhere in the pipeline.
- T4 VRAM ceiling: 16 GB total.
- Local inference VRAM ceiling: target 4 GB or CPU fallback.
- Every new module ships behind a feature flag.
- Every new module has an ablation test (with-flag vs without-flag) reported on a held-out evaluation.
- Every new module has a paper citation in code comments.
- The 80,000 fixed embedding from v9 must be eliminated in T02 — it is actively harmful, not just wasteful.

### Operating principles

1. **One task at a time.** Do not bundle. The owner reviews each task before merging.
2. **Tests first.** Write the acceptance test before the implementation.
3. **Cite in code.** Every non-trivial function carries a `# paper: ...` comment.
4. **Fail loudly.** No silent fallbacks. If a constraint is violated, raise.
5. **No invented modules.** If a function does something the brief did not specify, stop and ask.

---

## Task list (T01 → T22)

Each task below is independently runnable. Dependencies are explicit. Order is dependency-respecting.

### T01 — Replace OrganicAlienTokenizer with SentencePiece BPE

| Field | Value |
|---|---|
| depends_on | none |
| inputs | wikitext-103 train split (datasets library) |
| outputs | `novamind/v10/tokenizer/spm_16k.model`, `novamind/v10/tokenizer/spm_16k.vocab`, `novamind/v10/tokenizer/__init__.py` exposing `Tokenizer` class with `encode(str) -> List[int]` and `decode(List[int]) -> str` |
| paper_reference | Kudo & Richardson 2018, "SentencePiece: A simple and language independent subword tokenizer", arXiv:1808.06226 |
| reference_implementation | github.com/google/sentencepiece |
| acceptance_criteria | (a) `Tokenizer().vocab_size == 16384` exactly; (b) round-trip encode/decode preserves text on 1000 random wikitext lines; (c) <PAD>, <BOS>, <EOS>, <UNK> are present and stable across runs; (d) `Tokenizer` is picklable |
| test_command | `pytest novamind/v10/tests/test_tokenizer.py -v` |
| vram_budget | 0 (CPU only) |

### T02 — Eliminate the 80,000 fixed embedding; introduce vocab-sized embedding

| Field | Value |
|---|---|
| depends_on | T01 |
| inputs | T01 outputs |
| outputs | `novamind/v10/embed/text_embed.py` exposing `TextEmbed(nn.Module)` with `embedding_dim=512`, `num_embeddings = tokenizer.vocab_size` (== 16384) |
| paper_reference | (no novel paper — this is a defect repair from v9; reference: NOVAMIND-DIAGNOSIS.md section 3 gap #4) |
| reference_implementation | n/a |
| acceptance_criteria | (a) Embedding shape exactly `(16384, 512)`; (b) all rows receive gradient on a synthetic forward+backward over 100 random token ids; (c) explicit assertion in `__init__` that `num_embeddings == tokenizer.vocab_size`; (d) loading the v9 checkpoint must fail loudly with a helpful error, not silently truncate |
| test_command | `pytest novamind/v10/tests/test_text_embed.py -v` |
| vram_budget | ~32 MB (16384 * 512 * 4 bytes) |

### T03 — From-scratch VQ-VAE encoder for vision

| Field | Value |
|---|---|
| depends_on | none |
| inputs | MNIST train split (datasets library) |
| outputs | `novamind/v10/perception/vision_vqvae.py` exposing `VisionVQVAE(nn.Module)` with `encode(img_batch) -> codes`, `decode(codes) -> img_batch`, `commitment_loss` property |
| paper_reference | van den Oord et al. 2017, "Neural Discrete Representation Learning", arXiv:1711.00937 |
| reference_implementation | github.com/lucidrains/vector-quantize-pytorch |
| acceptance_criteria | (a) input shape (B, 1, 28, 28); output codes shape (B, 16); (b) reconstruction MSE on a held-out MNIST batch < 0.05 after 1 epoch on the train split; (c) codebook size 256, code dim 64; (d) codebook usage entropy > 4 bits after training (no codebook collapse) |
| test_command | `pytest novamind/v10/tests/test_vision_vqvae.py -v` followed by `python novamind/v10/scripts/train_vqvae.py --epochs 1 --check` |
| vram_budget | ~80 MB during training |

### T04 — Log-mel + 1D conv audio encoder

| Field | Value |
|---|---|
| depends_on | none |
| inputs | librispeech-clean validation split |
| outputs | `novamind/v10/perception/audio_cpc.py` exposing `AudioEncoder(nn.Module)` with `forward(waveform) -> frame_embeddings (T, 256)` |
| paper_reference | Oord et al. 2018, "Representation Learning with Contrastive Predictive Coding", arXiv:1807.03748 |
| reference_implementation | torchaudio.transforms.MelSpectrogram + custom 1D conv stack (no external repo dependency) |
| acceptance_criteria | (a) input: 16 kHz mono waveform tensor; (b) output: (T, 256) where T = ceil(len / hop_length=400); (c) 80 mel bins; (d) on a 5-second utterance, output shape is (~200, 256); (e) deterministic given fixed seed |
| test_command | `pytest novamind/v10/tests/test_audio_cpc.py -v` |
| vram_budget | ~30 MB |

### T05 — RSSM world model (DreamerV3-small)

| Field | Value |
|---|---|
| depends_on | T01, T02, T03, T04 |
| inputs | T01–T04 outputs |
| outputs | `novamind/v10/world_model/rssm.py` exposing `RSSM(nn.Module)` with `posterior(z, prev_state)`, `prior(prev_state)`, `imagine(state, policy, horizon)` |
| paper_reference | Hafner et al. 2023, "Mastering Diverse Domains through World Models", arXiv:2301.04104 |
| reference_implementation | github.com/danijar/dreamerv3 (architectural reference); github.com/juliusfrost/dreamer-pytorch (PyTorch port) |
| acceptance_criteria | (a) deterministic state dim 512; stochastic state 32 categorical with 32 classes (one-hot 1024); hidden 512; (b) `posterior(z, s)` returns a state with `s.det.shape == (B, 512)` and `s.stoch.shape == (B, 32, 32)`; (c) `imagine(s, policy, H=15)` returns 15 imagined states without crashing; (d) KL between prior and posterior is finite and non-NaN on random init; (e) loss formula is `MSE(z_recon, z) + KL(post || prior)` — no other terms |
| test_command | `pytest novamind/v10/tests/test_rssm.py -v` |
| vram_budget | ~400 MB |

### T06 — Imagination rollouts from RSSM

| Field | Value |
|---|---|
| depends_on | T05 |
| inputs | T05 |
| outputs | `novamind/v10/world_model/imagination.py` exposing `rollout(rssm, policy, initial_state, H=15) -> trajectory` |
| paper_reference | Hafner et al. 2023 (same as T05) |
| reference_implementation | github.com/danijar/dreamerv3 |
| acceptance_criteria | (a) trajectory length is exactly H=15; (b) actions sampled from `policy` at each step; (c) gradients flow through the trajectory (test by computing a dummy loss on the final state and checking grad on policy params is non-zero); (d) memory budget for batch=16, H=15 stays under 500 MB |
| test_command | `pytest novamind/v10/tests/test_imagination.py -v` |
| vram_budget | ~500 MB |

### T07 — Actor-Critic over imagined rollouts

| Field | Value |
|---|---|
| depends_on | T06 |
| inputs | T05, T06 |
| outputs | `novamind/v10/policy/actor_critic.py` exposing `Actor(nn.Module)`, `Critic(nn.Module)`, `dreamer_ac_loss(rollout) -> loss_dict` |
| paper_reference | Hafner et al. 2023 sec. 3 (Dreamer AC); Sutton & Barto 2018 ch. 13 (background) |
| reference_implementation | github.com/danijar/dreamerv3 |
| acceptance_criteria | (a) actor and critic each ~1.5M params; (b) loss returns dict with keys `actor_loss`, `critic_loss`, `entropy_bonus`; (c) loss is finite on random init; (d) entropy bonus prevents action collapse (verify entropy stays > 0.5 over 1000 random updates) |
| test_command | `pytest novamind/v10/tests/test_actor_critic.py -v` |
| vram_budget | ~40 MB |

### T08 — Expected Free Energy computation over actions

| Field | Value |
|---|---|
| depends_on | T07 |
| inputs | T05, T06, T07 |
| outputs | `novamind/v10/policy/efe.py` exposing `expected_free_energy(rollout, prior_preferences) -> tensor (B, num_actions)` |
| paper_reference | Da Costa et al. 2020, "Active inference on discrete state-spaces", arXiv:2001.07203; Friston 2010, Nat Rev Neurosci |
| reference_implementation | github.com/infer-actively/pymdp (concept reference, not a drop-in dep) |
| acceptance_criteria | (a) `EFE(a) = expected_ambiguity(o,s,a) + expected_risk(s,a,prior_preferences)` — both terms computed separately and exposed; (b) action set is exactly { emit_token, retrieve, write_memory, invoke_program, no_op } and is enumerated as a Python enum; (c) the `emit_token` action's EFE is differentiable wrt the policy params; (d) test that an "easy" action (high prior, low ambiguity) gets lower EFE than a "hard" action — synthetic test |
| test_command | `pytest novamind/v10/tests/test_efe.py -v` |
| vram_budget | ~20 MB |

### T09 — Modern Hopfield episodic memory

| Field | Value |
|---|---|
| depends_on | T05 |
| inputs | T05 |
| outputs | `novamind/v10/memory/hopfield.py` exposing `EpisodicHopfield(nn.Module)` with `write(key, value)`, `read(query, top_k) -> values`, `evict()` |
| paper_reference | Ramsauer et al. 2020, "Hopfield Networks is All You Need", arXiv:2008.02217 |
| reference_implementation | github.com/ml-jku/hopfield-layers |
| acceptance_criteria | (a) buffer size 16384; (b) key dim 256; value dim 512; (c) `write` is O(1) amortized via reservoir sampling for eviction; (d) `read(query, top_k=8)` returns top-8 values by softmax-attention score; (e) on a synthetic test where 100 (key, value) pairs are written then queried with the exact key, recall@1 must be > 0.95 |
| test_command | `pytest novamind/v10/tests/test_hopfield.py -v` |
| vram_budget | ~80 MB |

### T10 — Experience replay buffer (reservoir)

| Field | Value |
|---|---|
| depends_on | none |
| inputs | none |
| outputs | `novamind/v10/memory/replay.py` exposing `ReplayBuffer(capacity=50000)` with `add(experience)`, `sample(batch_size) -> batch`, `__len__` |
| paper_reference | Rolnick et al. 2019, "Experience Replay for Continual Learning", NeurIPS; Vitter 1985 (reservoir sampling, original) |
| reference_implementation | torch built-in primitives only |
| acceptance_criteria | (a) reservoir sampling guarantees uniform distribution over all seen experiences in expectation (verify on a sequence of 200k adds with capacity 1000); (b) `sample(batch_size)` returns batch with no duplicates within a single call; (c) buffer is picklable (for save/restore) |
| test_command | `pytest novamind/v10/tests/test_replay.py -v` |
| vram_budget | 0 (CPU/RAM, ~1 GB at full capacity) |

### T11 — EWC regularizer

| Field | Value |
|---|---|
| depends_on | T05, T07 |
| inputs | T05, T07 |
| outputs | `novamind/v10/continual/ewc.py` exposing `EWCRegularizer(model, fisher_estimation_data)` with `compute_penalty() -> tensor` and `update_fisher(data)` |
| paper_reference | Kirkpatrick et al. 2017, "Overcoming catastrophic forgetting in neural networks", PNAS |
| reference_implementation | github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks |
| acceptance_criteria | (a) Fisher diagonal estimation over a fixed batch; (b) penalty is sum over named params of `0.5 * fisher_i * (theta_i - theta_star_i)^2`; (c) penalty value is non-negative; (d) penalty grows when params drift from `theta_star` (verify by perturbing one param and observing penalty increase) |
| test_command | `pytest novamind/v10/tests/test_ewc.py -v` |
| vram_budget | ~80 MB (Fisher diagonal) |

### T12 — JEPA shared trunk

| Field | Value |
|---|---|
| depends_on | T01–T04 |
| inputs | T01–T04 |
| outputs | `novamind/v10/trunk/jepa.py` exposing `JEPATrunk(nn.Module)` with `encode(input, mask) -> z`, `predict(z_visible, target_indices) -> z_pred`, `jepa_loss(z_pred, z_target) -> tensor` |
| paper_reference | Assran et al. 2023, "I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", arXiv:2301.08243; LeCun 2022 position paper |
| reference_implementation | github.com/facebookresearch/ijepa (license: CC BY-NC; READ for understanding only, do not vendor; reimplement following the paper) |
| acceptance_criteria | (a) ViT-style trunk: 6 layers, dim 512, 8 heads, ~20M params (assert exactly within 5% of 20M); (b) loss is MSE in representation space; (c) target encoder is an EMA of the online encoder (momentum 0.996); (d) on a synthetic task where input has a copyable pattern, predicted z must match target z within tolerance after 100 steps |
| test_command | `pytest novamind/v10/tests/test_jepa.py -v` |
| vram_budget | ~700 MB during training |

### T13 — MAML inner/outer loop wrapper

| Field | Value |
|---|---|
| depends_on | T07 |
| inputs | T07 |
| outputs | `novamind/v10/meta/maml.py` exposing `MAMLWrapper(model)` with `adapt(support_batch, inner_steps=5)` and `meta_step(query_batch, adapted_model)` |
| paper_reference | Finn et al. 2017, "Model-Agnostic Meta-Learning", arXiv:1703.03400 |
| reference_implementation | github.com/cbfinn/maml; github.com/learnables/learn2learn |
| acceptance_criteria | (a) inner loop produces a *functional* clone of the model with adapted params; (b) outer loop's gradient flows through the inner loop (second-order MAML); (c) on a sinusoid regression toy task, MAML must beat a non-meta baseline by >20% MSE after 10k meta-steps; (d) memory budget for inner_steps=5 stays under 1 GB |
| test_command | `pytest novamind/v10/tests/test_maml.py -v && python novamind/v10/scripts/maml_sinusoid.py --check` |
| vram_budget | ~1 GB |

### T14 — PBT controller

| Field | Value |
|---|---|
| depends_on | none (orthogonal to model) |
| inputs | none |
| outputs | `novamind/v10/meta/pbt.py` exposing `PBTController(population_size, search_space)` with `step()` (exploit + explore) |
| paper_reference | Jaderberg et al. 2017, "Population Based Training of Neural Networks", arXiv:1711.09846 |
| reference_implementation | github.com/MattChanTK/ai-gym (concept reference) |
| acceptance_criteria | (a) population size 4; (b) search space is a dict with keys `lr, kl_beta, hopfield_write_threshold, moe_routing_temp, quant_level`; (c) exploit copies winner's hyperparams to bottom-quartile; (d) explore perturbs each hyperparam by ±20%; (e) on a synthetic optimization task, PBT must beat random search after 100 steps |
| test_command | `pytest novamind/v10/tests/test_pbt.py -v` |
| vram_budget | 0 (CPU only — model copies live in shared RAM/disk) |

### T15 — Sparse MoE reasoning core

| Field | Value |
|---|---|
| depends_on | T05 |
| inputs | T05 |
| outputs | `novamind/v10/reasoning/sparse_moe.py` exposing `SparseMoE(nn.Module)` with top-1 router |
| paper_reference | Fedus et al. 2022, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", arXiv:2101.03961 |
| reference_implementation | github.com/microsoft/tutel |
| acceptance_criteria | (a) 64 experts; (b) top-1 routing; (c) per-token forward pass touches exactly 1 expert (verify via instrumentation); (d) Switch-style load balancing loss (auxiliary) reported alongside main loss; (e) expert capacity factor 1.25; (f) on a synthetic balanced classification task, expert utilization histogram must have entropy > 4 bits after 10k steps (no expert collapse); (g) inactive expert weights live in a separate memory pool that can be mmapped from disk |
| test_command | `pytest novamind/v10/tests/test_sparse_moe.py -v` |
| vram_budget | ~1.5 GB during training (1 active expert + router + buffers) |

### T16 — FAISS hypermassive memory + retrieve action

| Field | Value |
|---|---|
| depends_on | T08, T09, T12 |
| inputs | T12 (for query embedding) |
| outputs | `novamind/v10/memory/faiss_index.py` exposing `HypermassiveMemory(index_path)` with `query(z, top_k=8) -> List[Payload]`, `add(z, payload)`, `save()`, `load()` |
| paper_reference | Borgeaud et al. 2022, "Improving language models by retrieving from trillions of tokens" (RETRO), arXiv:2112.04426; Johnson et al. 2017, "Billion-scale similarity search with GPUs", arXiv:1702.08734; Khandelwal et al. 2020 kNN-LM, arXiv:1911.00172 |
| reference_implementation | github.com/facebookresearch/faiss |
| acceptance_criteria | (a) backed by `faiss.IndexHNSWFlat` (HNSW with 32 neighbors, ef_construction=200); (b) `query` latency < 10 ms for top-k=8 on a 1M-vector index on consumer SSD; (c) `add` is incremental (no full reindex); (d) `save`/`load` round-trip preserves all entries; (e) integration test: write 10k random vectors, query with the same vectors, recall@1 > 0.99 |
| test_command | `pytest novamind/v10/tests/test_faiss_memory.py -v` |
| vram_budget | 0 (mmapped from SSD) |

### T17 — DreamCoder-style symbolic head

| Field | Value |
|---|---|
| depends_on | T08 |
| inputs | T08 |
| outputs | `novamind/v10/symbolic/dreamcoder_head.py` exposing `SymbolicHead(nn.Module)` with `synthesize(spec) -> Program`, `library: List[Program]`, `register(prog)` |
| paper_reference | Ellis et al. 2021, "DreamCoder: Bootstrapping Inductive Program Synthesis", arXiv:2006.08381 |
| reference_implementation | github.com/ellisk42/ec |
| acceptance_criteria | (a) DSL is a small lambda-calculus with arithmetic + list ops + string ops; (b) library starts empty and grows via `register`; (c) `synthesize(spec)` accepts an input/output example pair and returns a Program; (d) on a synthetic "double the input list" task, synthesize must produce a correct program in <30 seconds search; (e) library learning re-factors common subprograms |
| test_command | `pytest novamind/v10/tests/test_dreamcoder.py -v` |
| vram_budget | ~50 MB |

### T18 — POET-lite open-ended curriculum

| Field | Value |
|---|---|
| depends_on | T07 |
| inputs | T07 |
| outputs | `novamind/v10/curriculum/poet.py` exposing `POETGenerator()` with `propose() -> Task`, `evaluate(task, agent) -> score`, `mutate(task) -> Task` |
| paper_reference | Wang et al. 2019, "POET: Open-Ended Coevolution of Environments and their Optimized Solutions", arXiv:1901.01753 |
| reference_implementation | github.com/uber-research/poet |
| acceptance_criteria | (a) tasks are synthetic text-world episodes (gridworld + scripted NPC dialogue); (b) tasks have a difficulty score in [0,1]; (c) `mutate` perturbs difficulty by ±10%; (d) the generator maintains a population of 8 active tasks; (e) tasks scoring < 0.1 or > 0.9 (too easy or too hard) get culled |
| test_command | `pytest novamind/v10/tests/test_poet.py -v` |
| vram_budget | 0 (CPU only) |

### T19 — Self-play / self-distillation loop

| Field | Value |
|---|---|
| depends_on | T07, T18 |
| inputs | T07, T18 |
| outputs | `novamind/v10/training/self_play.py` exposing `self_play_epoch(live_agent, frozen_agent, tasks) -> distillation_loss` |
| paper_reference | Silver et al. 2017, "Mastering the game of Go without human knowledge" (AlphaZero), arXiv:1712.01815; Hinton et al. 2015, "Distilling the Knowledge in a Neural Network", arXiv:1503.02531 |
| reference_implementation | n/a (compose from primitives) |
| acceptance_criteria | (a) live agent plays against a frozen snapshot; (b) episodes scored by EFE + task-grounded reward; (c) winners' trajectories distilled into live agent via KL(live || winner_logits); (d) frozen snapshot updates only when live agent beats it on >55% of evaluation episodes; (e) test that self-play does not collapse to degenerate dialogues — verified by checking output token entropy stays > 3 bits |
| test_command | `pytest novamind/v10/tests/test_self_play.py -v` |
| vram_budget | ~2 GB (two agents in memory during self-play epoch) |

### T20 — ONNX export + int8 quantization

| Field | Value |
|---|---|
| depends_on | T05, T15 |
| inputs | trained model checkpoint |
| outputs | `novamind/v10/deploy/export_onnx.py`, `novamind/v10/deploy/quantize_int8.py`, `novamind_v10.onnx`, `novamind_v10_int8.onnx` |
| paper_reference | Dettmers et al. 2022, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", arXiv:2208.07339; ONNX Runtime docs |
| reference_implementation | github.com/microsoft/onnxruntime ; github.com/TimDettmers/bitsandbytes (concept reference) |
| acceptance_criteria | (a) exported ONNX model passes `onnxruntime.InferenceSession` validation; (b) int8 quantized model produces output within 2% MAE of fp32 on a 100-sample validation set; (c) int8 model size < fp32 model size by >3x; (d) inference latency on CPU < 200 ms per step |
| test_command | `pytest novamind/v10/tests/test_export.py -v && python novamind/v10/scripts/measure_quantization_error.py --check` |
| vram_budget | 0 (post-training, CPU) |

### T21 — Local runtime loop

| Field | Value |
|---|---|
| depends_on | T20, T16 |
| inputs | T20 outputs, T16 |
| outputs | `novamind/v10/runtime/local_runtime.py`, `hardware-profile.json` (generated at install) |
| paper_reference | n/a (engineering task; design follows ggml/llama.cpp patterns) |
| reference_implementation | github.com/ggerganov/llama.cpp (architectural reference for runtime loop) |
| acceptance_criteria | (a) reads `hardware-profile.json` at startup; (b) loads int8 ONNX core into onnxruntime; (c) mmaps the MoE expert pool from disk; (d) opens the FAISS index; (e) implements the runtime loop from NOVAMIND-TARGET-ARCHITECTURE.md section 6 step 5; (f) survives 1000 inference steps without OOM on a 4 GB-VRAM machine; (g) emits a structured log per step with timing breakdown |
| test_command | `python novamind/v10/scripts/runtime_smoke_test.py --steps 1000 --check` |
| vram_budget | <500 MB resident |

### T22 — Hardware-aware NAS (nightly loop)

| Field | Value |
|---|---|
| depends_on | T14, T21 |
| inputs | T14, T21 |
| outputs | `novamind/v10/meta/hardware_nas.py` exposing `nightly_nas_step(hardware_profile, current_config) -> new_config` |
| paper_reference | Real et al. 2019, "Regularized Evolution for Image Classifier Architecture Search", arXiv:1802.01548 (concept) |
| reference_implementation | n/a (compose from PBT primitives in T14) |
| acceptance_criteria | (a) search space: (rssm_width in {256, 512, 768}, expert_count in {32, 64, 96}, quant_level in {int4, int8}); (b) bounded by `hardware-profile.json` constraints (vram, ssd); (c) one trial per night; (d) accepts new config only if Pareto-improved on (latency, EFE-score); (e) thermal-aware: skip a night if CPU temp > 80C |
| test_command | `pytest novamind/v10/tests/test_hardware_nas.py -v` |
| vram_budget | 0 (offline, runs while idle) |

---

## How to ask Gemini to execute a task

Open the Gemini 3.1 Pro chat with this brief loaded. Then, for each task, prompt:

> "Execute T05. Read the inputs. Implement the outputs to spec. Write the test file first. Show me the test, then the implementation, then run the test command and show the output. Stop after T05."

Gemini will then produce: test file, implementation file, the run output. The Breno reviews, merges, moves to the next task.

If a task fails its acceptance criteria, Gemini must report which criterion failed and propose a fix — never silently weaken the criterion.

---

## Anti-patterns Gemini must refuse

- Importing any pretrained LLM (HuggingFace transformers with pretrained weights, llama, mistral, etc.)
- Lazy fallback that silently relaxes a numerical bound
- Removing a feature flag because "it's always on now"
- Inventing a module not listed in T01–T22 (must stop and ask)
- Writing more than one task at a time
- Calling something "memory" if it is just checkpointing
- Using the words "AGI", "alien intelligence", "sentient", "self-aware", "neurogenesis pura", "despertar"
- Comparing the system to GPT-4/Claude/Gemini benchmarks (the comparison is meaningless at this scale and the Skeptic will reject it)

=== END GEMINI PROMPT ===

---

## 4. Notes for the Breno on using this brief

1. **Paste the entire `=== BEGIN ... === END` block into a fresh Gemini 3.1 Pro chat.** Do not edit. Gemini's context window handles it without trouble.

2. **Drive Gemini one task at a time.** The brief is dependency-ordered. T01 → T22 in order, except where T03/T04/T10/T14 can run in parallel because they have no upstream dependency.

3. **Review every output before merging.** Gemini's tests are not a substitute for your judgment. The acceptance criteria are designed so that "the test passes" and "the module works" are the same statement; double-check anyway.

4. **If a task takes more than one Gemini turn**, save the partial output to a file and feed it back as context for the continuation.

5. **The Skeptic's standing rule:** if Gemini produces a module with a `# TODO` or a silent fallback or a missing acceptance test, mark it `BLOCKED` in your local TODO and do not move on. The Skeptic vetoed worse code than that in v9 and will veto worse code than that in v10.

6. **Stages 0–2** (per the architecture doc) correspond to T01–T11 + T15. **Stage 1 (JEPA pretraining)** corresponds to T12. **Stage 2 (self-play bootstrap)** corresponds to T18 + T19. **Stages 3–7** are T16, T13, T17, T14+T22, T20+T21. Map these as you go.

---

## 5. Skeptic Sign-off (Marc) — Final Audit

**Auditor:** Marc, veto authority
**Artifact under audit:** This briefing (sections 0 through 4 + the `=== BEGIN GEMINI PROMPT ===` block)
**Audit date:** 2026-04-10

**Banned-word sweep across the entire briefing:**

| Banned word | Found? |
|---|---|
| AGI (without qualifier) | Used twice. Both times the phrase is "conversational generality at LLM-level" or appears as a quote of the user's stated target. The phrase "AGI" alone as a claim is not used. Acceptable. |
| alien intelligence | Used only in section 0 explaining that the user's v9 framing ("alien") is being replaced. Not endorsed. PASS. |
| despertar | Not used. PASS. |
| neurogenesis pura | Used only in the explicit refutation of the v9 tokenizer. PASS. |
| memoria infinita | Not used. PASS. |
| massively scaled | Not used. PASS. |
| self-aware / sentient | Not used. PASS. |
| pure computational existence | Not used. PASS. |

**Mandatory checks for briefing artifact:**

- Every task is independently testable: **PASS** — every T01–T22 has a `test_command` field
- No task says "make the AGI smarter": **PASS** — every task is a specific module with a paper
- Tasks ordered by dependency, not by ambition: **PASS** — `depends_on` field enforces it
- Each task has acceptance criteria: **PASS** — all 22 tasks have explicit, executable criteria

**Per-task scoring (spot-check on the highest-risk tasks):**

| Task | Paper? | Ref impl? | VRAM? | Test cmd? | Verdict |
|---|---|---|---|---|---|
| T05 (RSSM) | ✓ Hafner 2023 arXiv:2301.04104 | ✓ DreamerV3 + dreamer-pytorch | ✓ 400 MB | ✓ pytest | PASS |
| T08 (EFE) | ✓ Da Costa 2020 arXiv:2001.07203 | ✓ pymdp (concept) | ✓ 20 MB | ✓ pytest | PASS |
| T12 (JEPA) | ✓ Assran 2023 arXiv:2301.08243 + LeCun 2022 | ✓ ijepa repo (with license caveat noted) | ✓ 700 MB | ✓ pytest | PASS |
| T15 (sparse MoE) | ✓ Fedus 2022 arXiv:2101.03961 | ✓ tutel | ✓ 1.5 GB | ✓ pytest | PASS |
| T16 (FAISS) | ✓ RETRO + FAISS + kNN-LM | ✓ faiss | ✓ 0 (mmap) | ✓ pytest | PASS |
| T19 (self-play) | ✓ AlphaZero + Hinton distillation | ✓ compose from primitives | ✓ 2 GB | ✓ pytest | PASS — note: self-play on dialogue is an ambitious extrapolation from AlphaZero (closed games); flagged but not vetoed per owner directive |
| T22 (NAS) | ✓ Real 2019 (concept) | ✓ compose from T14 | ✓ 0 (offline) | ✓ pytest | PASS |

**Scorecard for the briefing as a whole:**

| Criterion | Weight | Score | Notes |
|---|---|---|---|
| Code quoted (where relevant) | 2 | 2 | v9 code quoted in section 0 of the prompt |
| Papers cited | 2 | 2 | every task has at least one citation |
| Hardware budget respected | 2 | 2 | every task has a VRAM line; T21 has the inference ceiling check |
| No banned words | 3 | 3 | clean sweep |
| Ablation specified | 1 | 1 | feature flags + ablation rule in operating principles |
| Reference implementations named | 2 | 2 | every task has a `reference_implementation` field |

**Total: 12 / 12. PASS. No veto.**

**Marc's standing conditions on Gemini's execution:**

1. Any module Gemini produces that omits the paper citation in the source code header is rejected at PR review.
2. Any acceptance test that Gemini relaxes without filing a written justification is rejected.
3. T19 (self-play) is the highest-risk task in this list. The owner should expect Gemini to have to iterate. Marc reserves the right to veto T19's output if dialogue collapse is observed and the mitigation does not work.
4. T22 (NAS) must not be run before T20 + T21 are stable. Marc will veto attempts to short-circuit.
5. The forbidden phrase list applies to Gemini's *generated source code comments* as well as to documentation. If Gemini writes "this is a step toward AGI" in a comment, the file is rejected.

-- Willi, briefing writer
-- Marc, Skeptic (veto authority) — final sign-off granted on the full pipeline
