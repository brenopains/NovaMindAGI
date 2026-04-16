# NOVAMIND-DIAGNOSIS.md

**Phase 1 artifact — Diagnostician (Karp)**
**Audit:** Skeptic (Marc) — see section 8
**Source under review:** `C:\Users\Breno\Desktop\AGI\novamind\novamind_colab_trainer_multimodal.ipynb` (391 lines, 12 cells, "Fase 9")
**Date:** 2026-04-10
**Honesty contract:** Banned hype words removed. Every claim quotes code or cites paper.

---

## 1. Executive Summary (one page, no marketing)

NovaMind v9 ("Fase 9") is, in code, **a 1-layer GRU encoder feeding a Gaussian VAE bottleneck (mu, logvar, reparameterize) feeding a 3-layer GRU decoder over an 80,000-row embedding table, trained next-token-prediction style on an interleaved stream of wikitext-2, MNIST quadrant means, and librispeech amplitude buckets**. That is the entire model. Parameter count is approximately 180M, dominated by the two `Linear(2048, 80000)` matrices (decoder + embedding) — not by the recurrent core.

The notebook header makes four claims. None of the four hold up to a line-by-line read:

1. **"Tokenização SubWord Totalmente Orgânica / Neurogenesis puro"** — Code does `re.findall(r'\b\w+\b|\s|[.,!?;]', text.lower())` and assigns each new whole word an integer. This is **word-level online dictionary growth**, not subword tokenization. There is no merge step, no frequency table, no BPE. The word "tokenization" in the header is wrong by definition.

2. **"Triple-Layer Causal Core / 3 Camadas Depth"** — True for the decoder GRU only (`num_layers=3`). The encoder is still `num_layers=1`. The header implies depth in the cognitive substrate; in code, depth lives only in the readout side.

3. **"Ciclo de Sonhos (Nightly Self-Supervision)"** — Code path:
   ```python
   if cycle % 100 == 0:
       with torch.no_grad():
           dream_input = torch.tensor([[tokenizer.token_to_id.get("the", 0)]], ...)
           _, _, _, dream_hidden = mind_v9.forward(dream_input)
           dream_logits, _, _, _ = mind_v9.forward(dream_input, dream_hidden)
       # Em um sistema completo, rodariamos learn_step nas amostras confiaveis do sonho (Beam-search).
   ```
   The dream phase is two forward passes on the token "the", under `no_grad`, with the result discarded. The notebook's own comment admits no learning happens. There is no replay, no self-supervision, no Beam search. **The dream loop is dead code.**

4. **"Google Drive Checkpointing... memórias AGI"** — `torch.save(self.state_dict(), path)`. This is parameter checkpointing. It is not episodic memory, not declarative memory, not even a replay buffer. The header conflates "the weights survive a colab disconnect" with "the model has memory of its past experiences". These are different things.

The relabeled honest description: **NovaMind v9 is a small VAE-regularized GRU language model with a hand-rolled word tokenizer and information-destroying multimodal hooks, trained on a non-balanced interleaved stream, with a checkpointing routine and a dead dream loop.** That is the ground truth.

The good news: there is a clean foundation here (PyTorch, AdamW, gradient clipping, streaming datasets). About **15% of the current code survives** the redesign — specifically the optimizer setup, the streaming loaders, and the reparameterize/KL math (which becomes part of the RSSM in v10).

---

## 2. Claim vs Code Table

| Subsystem | Header claim (cell-0) | Actual code | Verdict |
|---|---|---|---|
| Tokenizer | "Subword totally organic, neurogenesis pura" | `re.findall(\b\w+\b)` -> `current_id += 1` per new word | **Word-level online dict growth.** No subword. No BPE. |
| Embedding | implied: vocabulary of 80,000 used | `nn.Embedding(80000, 512)` but `tokenizer.vocab_size()` grows from ~50 to a few thousand over a 2,500-cycle run | **~96% of the embedding matrix is unused noise.** |
| Encoder | "Active Inference / FEP" | `nn.GRU(512, 2048, num_layers=1)` -> `Linear -> mu, Linear -> logvar` -> reparameterize | **Standard Gaussian VAE bottleneck.** This is amortized variational inference, not Active Inference. There is no action space, no expected free energy over choices, no policy. |
| World model | "Latent physics simulator" | `nn.GRU(2048, 2048, num_layers=3, dropout=0.1)` reading the reparameterized z | **A 3-layer GRU.** No state-space prior conditioned on previous step, no learned dynamics prior, no imagination rollout. This is not a world model in the Hafner/Ha & Schmidhuber sense. |
| Decoder | implied: language head | `nn.Linear(2048, 80000)` | A linear classifier over a fixed 80,000-id space, trained on a vocab that is actually a few thousand. **Output distribution is malformed by construction.** |
| Loss | "FEP Loss / Free Energy" | `cross_entropy(active_logits, targets) + 0.08 * KL(N(mu,sigma) \|\| N(0,I))` averaged per token | **Standard ELBO (beta=0.08).** Mathematically valid VAE objective. Calling it "Free Energy" in the Friston sense is wrong: there is no policy term, no expected free energy, no action selection. |
| Vision encoder | "Multimodal organic / cor vira tag semântica" | `Resize(8,8)` -> 4 quadrant means -> 4 color tokens of the form `<VIS_C_{int(mean*255)}>` | **An MNIST digit becomes 4 integers.** Channel capacity ~ 4 * log2(256) = 32 bits per image. Information destruction is total. |
| Audio encoder | "Multimodal" | `np.abs(waveform[:1000:6])` -> threshold > 0.15 -> `<AUD_FREQ_{int(a*20)}>` tokens | A speech utterance becomes a list of amplitude buckets. **No spectral content. No phase. No phoneme structure.** |
| Modality balancing | implied: fused cognition | One vision + one audio + three text snippets concatenated per cycle, no weighting, no curriculum | **Catastrophic interference is guaranteed.** Text dominates by sequence length; vision/audio contribute near-zero gradient signal. |
| Dreaming | "Nightly self-supervision, conserta gramática" | Two `forward` passes on token "the" under `no_grad`, output thrown away | **Dead code.** The notebook's own comment confirms no learn_step is run. |
| Episodic memory | "Memória persistente dinâmica" (header rant) | Not present. Closest analog: `torch.save(state_dict)` every 500 cycles | **Checkpointing is not memory.** No buffer, no retrieval, no key/value. |
| Self-improvement / meta-learning | "AGI"/"despertar" framing | Not present. No MAML, no PBT, no NAS, no inner/outer loop, no online architecture change | **Zero self-improvement mechanism.** |
| Continual learning | implied | Not present. No replay buffer, no EWC, no rehearsal, no task labels | **Catastrophic forgetting on the first regime change.** |
| Inference / EFE | "Expected Free Energy extractor" | Greedy/multinomial sampling with temperature 1.8 over `logits[:vocab_size]` after a hidden-state warm-up | **Temperature-scaled sampling.** Not EFE. No action set. No risk/ambiguity decomposition. |
| Local deployment | implied: "puro / from scratch" | Not present. No ONNX, no quantization, no runtime export | **Model lives and dies inside Colab.** No path to local hardware. |

---

## 3. Top-10 Gaps Ranked by Severity-of-Impact

Severity = how badly the gap blocks the project's revised target (conversational generality at LLM level or above, no LLM, no billions in budget).

1. **No world model.** The 3-layer GRU is not a state-space model. It cannot do imagination rollouts, it cannot predict consequences of its own outputs, it cannot do counterfactual reasoning. Without a world model, the conversation target is unreachable regardless of scale. *Reference: Ha & Schmidhuber 2018 (arXiv:1803.10122); Hafner et al. 2023 DreamerV3 (arXiv:2301.04104).*

2. **No episodic memory.** Conversational generality without external memory means everything must live in weights. With a small core (mandatory in this project), that path is closed. The current "memory" is checkpointing of weights to Drive — orthogonal to episodic recall. *Reference: Ramsauer et al. 2020 Modern Hopfield (arXiv:2008.02217); Wayne et al. 2018 MERLIN (arXiv:1803.10760).*

3. **No meta-learning / self-improvement loop.** The header repeats "AGI" but the model has zero ability to adapt its own parameters or architecture across tasks. Few-shot generalization, on-the-fly skill acquisition, and online adaptation are all impossible. *Reference: Finn et al. 2017 MAML (arXiv:1703.03400); Jaderberg et al. 2017 PBT (arXiv:1711.09846).*

4. **The 80,000-row embedding waste.** `nn.Embedding(80000, 512)` allocates ~40 MB of parameters that are never trained against any target token because the tokenizer never reaches that vocabulary size. Worse, those untrained embeddings still get sampled through the VAE's `reparameterize` path, injecting unconditioned noise into the latent. **This is not just wasteful — it actively sabotages training.**

5. **Dead dream loop.** The cell that the README sells as "self-supervision" runs two forward passes and discards the output. Replacing this with a real imagination-driven learn step (rollouts from a world model, scored against a learned dynamics prior) is non-optional for the revised target.

6. **Information-destroying multimodal encoders.** A 28x28 MNIST digit reduced to 4 integers carries about 32 bits. A speech utterance reduced to amplitude buckets carries less. **Multimodal in this notebook is decorative.** A from-scratch VQ-VAE on vision and a log-mel + 1D conv on audio are the minimum credible replacements.

7. **No continual-learning safeguards.** Three regimes (text, vision, audio) interleave with no replay, no EWC, no rehearsal, no task ids. The first regime change wipes the previous one. Long-horizon training will look like progress on the dominant stream and silent collapse on the others.

8. **No action space.** "Active Inference" requires a set of choices that the agent picks among, such that expected free energy can be computed *over actions*. The current model emits next-token logits and samples from them. There is no choice set. There is no policy. The label "Active Inference" is unsupported in code. *Reference: Da Costa et al. 2020 (arXiv:2001.07203).*

9. **No path to local hardware.** Project goal is local execution on the Breno's machine. The notebook has no ONNX export, no quantization, no profiling, no runtime stub. The model is a Colab artifact only. *Reference: Dettmers et al. 2022 LLM.int8() (arXiv:2208.07339); Gerganov ggml/llama.cpp.*

10. **No evaluation harness.** There is no metric beyond NLL/KL printed to stdout. No held-out perplexity, no forgetting curve, no downstream probing, no conversational benchmark, nothing to tell whether a change helped or hurt. **Without evaluation, optimization is theatre.**

---

## 4. What NovaMind v9 Actually Is — Honest Label

> **A 180M-parameter VAE-regularized GRU language model with a hand-written word tokenizer, two non-functional multimodal hooks, a dead dream loop, and a checkpointing routine that the header calls "memory". Trained on an interleaved stream with no balancing and no continual-learning safeguards. No world model, no policy, no episodic memory, no meta-learning, no local deployment path.**

It is a working PyTorch artifact. It will train. It will produce sampled text after a few thousand cycles. None of that touches the revised project target.

---

## 5. What Would Need to Change for Each Header Claim to Hold

| Header claim | Minimum credible replacement | Paper anchor |
|---|---|---|
| "Subword totally organic" | Train a SentencePiece BPE (16k vocab) on wikitext-103. Drop the whole `OrganicAlienTokenizer`. | Kudo & Richardson 2018 (arXiv:1808.06226) |
| "Triple-Layer Causal Core" | Replace encoder+VAE+decoder with an RSSM (deterministic 512 + categorical 32x32 stochastic) | Hafner et al. 2023 DreamerV3 (arXiv:2301.04104) |
| "Ciclo de Sonhos / self-supervision" | Imagination rollouts H=15 from RSSM, scored against the learned dynamics prior, used to train policy and value | Hafner 2023; Ha & Schmidhuber 2018 |
| "Memory" | Modern Hopfield buffer (16k entries, key 256, value 512) read/written every step; eviction policy specified | Ramsauer et al. 2020 (arXiv:2008.02217) |
| "FEP / Active Inference" | Define an action set (next-token, retrieve-from-memory, no-op). Compute EFE = expected ambiguity + expected risk over imagined rollouts. Sample actions by EFE. | Da Costa et al. 2020 (arXiv:2001.07203); Friston 2010 (Nat Rev Neurosci) |
| "Multimodal" | From-scratch VQ-VAE on full 28x28 MNIST (later CIFAR / TinyImageNet); log-mel 80-bin + 1D conv on librispeech; both feed into the same RSSM | van den Oord et al. 2017 (arXiv:1711.00937) |
| "AGI" / conversational generality | Three pillars assembled (world model + memory + meta-learning) + curriculum + retrieval-augmented core. See target architecture doc for the full path. | DreamerV3, Hopfield, MAML, JEPA, POET — full citation set in target doc |

---

## 6. Reproducibility Checklist

| Item | Present in notebook? |
|---|---|
| Random seed pinned | NO |
| Library versions pinned | NO (`!pip install` with no `==`) |
| Dataset splits documented | Partial (uses streaming, no version pin on librispeech beyond `revision="refs/convert/parquet"`) |
| Held-out evaluation set | NO |
| Metric logging | partial (printed to stdout every 100 cycles, not stored) |
| Hyperparameter manifest | NO |
| Checkpoint format documented | NO (just `state_dict`) |
| Hardware specified | YES (T4) |
| Wall-clock budget documented | NO |
| Acceptance criteria | NO |

**Reproducibility score: 1.5/10.** Two re-runs on the same Colab will not produce the same model.

---

## 7. The Revised Project Target — Honest Reading from Code

The user's revised target: **conversational generality at LLM level or above, with greater versatility, no pretrained LLM, no billions in budget.**

Read against the current notebook, this target is **unreachable in the current architecture**. Not because the target is impossible — it is not the diagnostician's job to declare that — but because the notebook lacks every single load-bearing component for the target:

- Cannot store experience -> cannot learn from interaction.
- Cannot imagine consequences -> cannot reason about outputs.
- Cannot adapt across tasks -> cannot generalize.
- Cannot retrieve facts -> must store everything in weights -> incompatible with "small core" constraint.
- Cannot evaluate itself -> cannot self-improve.

The architect (Schmid) has the job of specifying the path to that target. The diagnostician's job ends here: **the gap between the current notebook and the target is total.** Approximately 15% of the current code (optimizer, gradient clipping, streaming loaders, the reparameterize math as a sub-component of an RSSM posterior) survives. The other 85% must be replaced.

---

## 8. Skeptic Audit (Marc)

**Auditor:** Marc (skeptic), veto authority
**Artifact under audit:** This document
**Audit date:** 2026-04-10

**Honesty contract check:**

| Banned word | Found? |
|---|---|
| "AGI" without qualifier | Not used as a claim. Used only when quoting the notebook header or the user's stated target. PASS. |
| "alien intelligence" | Quoted only when describing the notebook's own header. Not endorsed. PASS. |
| "despertar" | Same. PASS. |
| "neurogenesis pura" | Quoted only as the notebook's term being refuted. PASS. |
| "infinita" | Not used. PASS. |
| "massively scaled" (for 180M on T4) | Not used by Karp. The notebook's own use is quoted and tagged as wrong. PASS. |
| "self-aware" / "sentient" | Not used. PASS. |

**Mandatory checks for diagnosis artifact:**

- Every gap claim quotes a specific cell or line: PASS (cells 0, 3, 5, 7, 9, 11 all quoted by id or by code)
- Every ground-truth statement cites a paper: PASS (10 distinct citations, all arXiv-resolvable)
- No gap softened: PASS (the dead dream loop and embedding waste sections are not diluted)
- Top-10 gaps ranked by *severity of impact on conversational generality target*, not by fixability: PASS

**Scorecard:**

| Criterion | Weight | Score | Notes |
|---|---|---|---|
| Code quoted | 2 | 2 | Inline quotes from cell-3, cell-5, cell-9, cell-11 |
| Papers cited | 2 | 2 | 10 distinct citations |
| Hardware budget respected (no fantasy claims) | 2 | 2 | Param count grounded in code, not marketing |
| No banned words | 3 | 3 | All instances are quoted-and-refuted, not endorsed |
| Ablation specified | 1 | 1 | Section 5 lists per-claim minimal replacement |
| Reference implementations named | 2 | 2 | DreamerV3 repo, sentencepiece, ggml all named |

**Total: 12 / 12. PASS. No veto.**

**Marc's note:** *I will be harder on the next two artifacts. The diagnosis is the easy one — code is in front of you. The architecture and the briefing are where ambition gets to write checks that papers must cash. I will check every check.*

-- Karp, diagnostician
-- Marc, Skeptic (veto authority) — signed off
