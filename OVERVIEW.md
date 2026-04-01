# Karpathy's Autoresearch — Complete Project Overview

> "One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun… That era is long gone." — @karpathy, March 2026

---

## 1. What is autoresearch?

**autoresearch** is an autonomous AI research system. You give an AI agent a real, working LLM training setup and let it run experiments overnight—on its own. The agent modifies code, trains for 5 minutes, checks whether the result improved, keeps or discards the change, and loops forever. You wake up to a log of experiments and (hopefully) a better model.

The core philosophy is:

- **You don't touch Python files.** Instead, you write `program.md`—a Markdown "skill file" that tells the agent what to do.
- **One file to modify.** The agent edits only `train.py`. Everything else is fixed.
- **One metric.** `val_bpb` (validation bits per byte). Lower is better.
- **One fixed time budget.** Every run trains for exactly 5 minutes (wall clock). This makes all experiments directly comparable regardless of what the agent changes.

---

## 2. Repository layout

```
autoresearch/
├── prepare.py     ← Fixed constants, data download, tokenizer training, dataloader, evaluation
├── train.py       ← The ONE file the agent edits: model, optimizer, hyperparameters, training loop
├── program.md     ← Agent instructions ("the research org code" written by the human)
├── pyproject.toml ← Python dependencies (uv project)
└── OVERVIEW.md    ← This file
```

---

## 3. `prepare.py` — Fixed infrastructure (do not modify)

This file is intentionally **read-only**. It contains everything that must remain constant across all experiments so that results are comparable.

### 3.1 Constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_SEQ_LEN` | 2048 | Token context length used for both training and evaluation |
| `TIME_BUDGET` | 300 s | Fixed 5-minute training wall-clock budget |
| `EVAL_TOKENS` | ~20 M | Number of tokens evaluated on the validation set |
| `VOCAB_SIZE` | 8192 | BPE vocabulary size |
| `VAL_SHARD` | 6542 | Pinned validation shard (always the same data) |

### 3.2 Data download

Training data comes from the [`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) dataset on HuggingFace, stored as Parquet shards. The download function (`download_data`) fetches shards in parallel and retries on failure. Shard 6542 is always reserved for validation.

### 3.3 Tokenizer training

A **BPE tokenizer** is trained using [`rustbpe`](https://github.com/karpathy/rustbpe) (a Rust-accelerated implementation) on up to 1 GB of text from the training shards. It uses a GPT-4-style split pattern. After training, the vocabulary is wrapped in a `tiktoken.Encoding` object and serialized to `~/.cache/autoresearch/tokenizer/tokenizer.pkl`. A companion `token_bytes.pt` file records the byte-length of every token (used for the BPB metric).

### 3.4 Dataloader (`make_dataloader`)

A **BOS-aligned, best-fit-packing** dataloader:

1. Documents are tokenized in batches from Parquet row-groups.
2. A buffer of ~1000 tokenized documents is maintained.
3. Each row of a batch starts with a `<BOS>` token and is filled to length `T+1` using best-fit bin-packing (find the largest document that fits, and crop the shortest one only when nothing fits).
4. This achieves **100% utilization** (no padding waste).
5. The resulting `(B, T)` input and target tensors are pinned to CPU and asynchronously transferred to GPU.

### 3.5 Evaluation (`evaluate_bpb`)

The fixed evaluation metric is **bits per byte (BPB)**:

```
BPB = Σ(cross_entropy_nats) / (log(2) × Σ(target_bytes))
```

Special tokens (byte length 0) are excluded. Using bytes instead of tokens makes the metric **vocabulary-size-independent**, so architectural changes that alter the vocabulary are fairly compared. Lower BPB = better compression = better language model.

---

## 4. `train.py` — The agent's sandbox

This is the one file the agent modifies. It contains the full GPT model definition, the optimizer, all hyperparameters, and the training loop. Everything is fair game.

### 4.1 Model architecture — `GPT`

The model is a transformer language model with several modern modifications:

#### Token and value embeddings

- **`wte`** — standard token embedding table `(vocab_size × n_embd)`.
- **Value embeddings** — a second, separate embedding table `(vocab_size × kv_dim)` used by alternating layers (see ResFormer below). Initialized uniformly and cast to `bfloat16`.

#### Rotary Position Embeddings (RoPE)

Position information is injected via **RoPE** applied to queries and keys. The cosine/sine tables are precomputed for `sequence_len × 10` positions (to support longer sequences at inference) and stored as non-persistent buffers.

#### RMS normalization

`F.rms_norm` is used throughout instead of LayerNorm — no learned scale or bias, just normalization.

#### QK normalization

After computing Q and K projections, both are passed through RMS norm before the attention dot product. This stabilizes training at large model sizes.

#### Sliding window attention (`SSSL` pattern)

Layers alternate between:
- **S** (short window) — attends to only the nearest `T/2` tokens (local context)
- **L** (long window) — full causal attention over all `T` tokens

The last layer is always forced to be `L` (full attention). This reduces computation in early layers while keeping global context in the final layer.

#### Value residual (ResFormer)

In alternating layers (those with `has_ve=True`), the value vector is augmented:

```
v = v_projected + sigmoid_gate(x[:, :32]) * v_embedded
```

The gate is a small linear layer that reads the first 32 channels of the hidden state and produces a per-head scalar. This is initialized to zero so the gate starts neutral (≈1.0 after the `2 × sigmoid` scaling). This technique (from the ResFormer paper) helps the model retain information across layers.

#### Residual stream with per-layer scalars

The forward pass uses a **learnable residual mixing** strategy:

```python
x = resid_lambda[i] * x + x0_lambda[i] * x0
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

`resid_lambdas` (initialized to 1.0) and `x0_lambdas` (initialized to 0.1) are learned per-layer scalars that control how much of the previous residual stream and the original embedding (`x0`) are retained.

#### MLP with squared ReLU

```python
x = relu(W_fc @ x)²
x = W_proj @ x
```

Expansion ratio of 4×. Squared ReLU (ReGLU-free, simpler) acts as a smooth activation with better gradient flow.

#### Output logit softcapping

```python
logits = 15 * tanh(logits / 15)
```

Caps logit magnitudes at ±15 to prevent loss spikes from extreme logit values.

#### Default model size (DEPTH=8)

With `DEPTH=8` and `ASPECT_RATIO=64`:
- `model_dim = 8 × 64 = 512`, rounded up to nearest multiple of `HEAD_DIM=128` → **512**
- `num_heads = 512 / 128 = 4`
- Approximately **50 M parameters**

### 4.2 Optimizer — `MuonAdamW`

A combined optimizer that uses different update rules for different parameter types:

| Parameter type | Optimizer | Default LR |
|---|---|---|
| Token embeddings (`wte`) | AdamW | 0.6 |
| Value embeddings | AdamW | 0.6 |
| LM head (unembedding) | AdamW | 0.004 |
| Per-layer scalars (`resid_lambdas`) | AdamW | `0.5 × 0.01` |
| Per-layer scalars (`x0_lambdas`) | AdamW | 0.5 |
| All 2D matrix params (attention, MLP) | **Muon** | 0.04 |

All AdamW learning rates are scaled by `1/√(model_dim/768)` to remain well-tuned across different model sizes.

#### Muon optimizer

Muon replaces the Adam update for weight matrices with an **orthogonalized gradient** update:

1. **Nesterov momentum** — standard lookahead momentum with `momentum=0.95`.
2. **Polar express orthogonalization** — iteratively applies Newton-Schulz iterations (`ns_steps=5`) using pre-computed polynomial coefficients to project the gradient onto the manifold of orthogonal matrices. This is equivalent to computing the matrix sign function (left/right polar factor).
3. **NorMuon variance reduction** — scales each gradient element by the reciprocal square root of an exponential moving average of its squared value (similar to AdaGrad/RMSProp), but operating on the already-orthogonalized gradient.
4. **Cautious weight decay** — applies weight decay only to parameters whose gradient and value have the same sign, preventing the optimizer from decaying parameters it is actively trying to increase.

LR for Muon is further scaled by `√max(1, rows/cols)` to account for rectangular matrix shapes.

Both `adamw_step_fused` and `muon_step_fused` are compiled with `@torch.compile` for maximum GPU efficiency.

### 4.3 Hyperparameters

```python
ASPECT_RATIO = 64        # model_dim = depth × ASPECT_RATIO
HEAD_DIM = 128           # target attention head dimension
WINDOW_PATTERN = "SSSL"  # layer attention window pattern
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per gradient step
DEPTH = 8                # number of transformer layers
DEVICE_BATCH_SIZE = 128  # micro-batch size (reduce if OOM)
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0       # no warmup by default
WARMDOWN_RATIO = 0.5     # cosine cooldown for last 50% of training
FINAL_LR_FRAC = 0.0      # LR decays to zero
```

Gradient accumulation is computed automatically: `grad_accum_steps = TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE × MAX_SEQ_LEN)`.

### 4.4 Training loop

1. **Compilation** — `torch.compile(model, dynamic=False)` traces the model for maximum throughput.
2. **Micro-step loop** — accumulates gradients over `grad_accum_steps` micro-batches.
3. **Schedule** — learning rate and Muon momentum are updated every step based on `progress = total_training_time / TIME_BUDGET`.
   - LR: flat → cosine cooldown to 0 over the last 50%.
   - Muon momentum: linearly ramps from 0.85 → 0.95 over the first 300 steps.
   - Weight decay: linearly decays from `WEIGHT_DECAY` → 0 as training progresses.
4. **Fast-fail** — if loss is NaN or > 100 the script exits immediately with `FAIL`.
5. **GC management** — Python's garbage collector is frozen after step 0 to avoid periodic ~500ms GC stalls.
6. **Time budget** — the loop breaks as soon as `total_training_time >= TIME_BUDGET` (after the first 10 warmup steps used for `torch.compile` tracing).
7. **Evaluation** — after training ends, `evaluate_bpb` runs the fixed validation pass.

### 4.5 Output format

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

`mfu_percent` is model FLOPs utilization — the ratio of actual FLOPs/sec to the theoretical peak of an H100 in BF16.

---

## 5. `program.md` — The agent's instructions

This Markdown file acts as a **lightweight "skill"** for the AI agent. The human edits this file to customize the research strategy. The default version defines:

- **Setup** — how to create a branch, read the codebase, and initialize `results.tsv`.
- **Experimentation rules** — what the agent CAN (modify `train.py`) and CANNOT do (modify `prepare.py`, install packages, change the evaluation harness).
- **The experiment loop** — an infinite loop: read git state → modify `train.py` → commit → run → read results → log → keep or discard → repeat.
- **Logging format** — `results.tsv` with columns: `commit`, `val_bpb`, `memory_gb`, `status`, `description`.
- **Autonomy requirement** — the agent must NEVER stop or ask for confirmation. It runs until manually interrupted.

---

## 6. End-to-end flow

```
Human
  │
  ├─ edits program.md  ← "research org code"
  │
  └─ starts AI agent ──► reads program.md
                              │
                         ┌────▼─────────────────────────────────────────┐
                         │  SETUP                                        │
                         │  1. git checkout -b autoresearch/<tag>        │
                         │  2. read README.md, prepare.py, train.py      │
                         │  3. create results.tsv                        │
                         └────┬─────────────────────────────────────────┘
                              │
                         ┌────▼─────────────────────────────────────────┐
                         │  LOOP FOREVER                                 │
                         │  1. devise experiment idea                    │
                         │  2. edit train.py                             │
                         │  3. git commit                                │
                         │  4. uv run train.py > run.log 2>&1            │
                         │  5. grep val_bpb from run.log                 │
                         │  6. append row to results.tsv                 │
                         │  7. if improved → keep commit                 │
                         │     else        → git reset --hard HEAD~1     │
                         └──────────────────────────────────────────────┘
```

---

## 7. Key design decisions

| Decision | Rationale |
|---|---|
| **Fixed 5-minute time budget** | All experiments are directly comparable across architecture/hyperparameter changes. Eliminates the need to control for compute. |
| **One file to modify (`train.py`)** | Keeps the agent's scope manageable; diffs are human-reviewable. |
| **`val_bpb` as metric** | Vocabulary-size-independent → fair comparison even if vocab size changes. |
| **Single GPU** | Self-contained, no distributed complexity. |
| **`program.md` as "research org code"** | The human iterates on the agent's strategy at a meta level, while the agent iterates on the model. |
| **BOS-aligned best-fit packing** | 100% token utilization; each sequence is a real document boundary, preserving natural language structure. |
| **Muon for matrices, AdamW for embeddings** | Matrix parameters benefit from orthogonalized updates (better conditioning); embeddings are sparse and suit Adam's per-parameter adaptivity. |

---

## 8. Quick start

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

# Download data shards and train tokenizer (~2 min, one-time)
uv run prepare.py

# Run a single 5-minute training experiment
uv run train.py

# Start autonomous agent (e.g. Claude/Codex in this repo)
# Prompt: "Have a look at program.md and let's kick off a new experiment!"
```

**Requirements:** Single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

---

## 9. Smaller compute platforms

For non-H100 setups, the recommended tuning knobs (in order of impact) are:

1. Use a lower-entropy dataset (e.g. [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)).
2. Reduce `MAX_SEQ_LEN` in `prepare.py` (e.g. 256–512).
3. Reduce `DEPTH` in `train.py` (e.g. 4 instead of 8).
4. Reduce `VOCAB_SIZE` (e.g. 4096, 2048, or byte-level 256).
5. Reduce `TOTAL_BATCH_SIZE` (keep as power of 2, e.g. `2**14`).
6. Use `WINDOW_PATTERN = "L"` (skip sliding windows).

---

## 10. Dependencies

| Package | Role |
|---|---|
| `torch 2.9.1` (cu128) | Deep learning framework |
| `kernels` | Flash Attention 3 kernel loader |
| `rustbpe` | Fast BPE tokenizer training in Rust |
| `tiktoken` | Tokenizer runtime |
| `pyarrow` | Parquet data reading |
| `requests` | Data shard downloading |
| `numpy`, `pandas`, `matplotlib` | Data analysis / plotting (used in `analysis.ipynb`) |
