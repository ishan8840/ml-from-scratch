# nanoGPT

A minimal, from-scratch implementation of a decoder-only Transformer in PyTorch, built to learn how *Attention Is All You Need* actually works under the hood. Trains a small character-level language model on any plain-text file and generates samples from it.

No `nn.MultiheadAttention`, no `F.scaled_dot_product_attention` — every piece of attention is written out by hand so the math stays visible.

## What's implemented

- **Multi-head self-attention from scratch.** Each `Head` computes its own Q/K/V projections, scaled dot-product attention, and causal masking via a registered lower-triangular buffer. `MultiHeadAttention` runs the heads in a `ModuleList` and concatenates their outputs.
- **Pre-norm Transformer blocks** with residual connections around attention and a 4×-expansion feedforward (`Linear → ReLU → Linear`).
- **Learned token + positional embeddings** summed at the input.
- **KV caching for generation.** During autoregressive decoding, each head caches its past keys and values so subsequent steps only run the model on the single new token. The causal mask is sliced based on the current key length (`tril[T_k - T_q : T_k, :T_k]`) so cached and new positions are masked correctly.
- **Context-window fallback.** When generation runs past `chunk_size` tokens, the cache is dropped and the model recomputes from the truncated tail.
- **Temperature sampling** (with `temperature=0` handled as greedy argmax).
- **Attention visualization** — `visualize_attention.py` exposes the per-head attention weights from every layer for a generated sequence.

## Repo layout

```
config.py              # Hyperparameters (dataclass)
data.py                # Char-level tokenizer + batch sampling
model.py               # Head, MultiHeadAttention, Block, GPT
train.py               # Training loop with periodic eval
generate.py            # Load a checkpoint and sample from it
visualize_attention.py # Attention weight visualization
```

## Quickstart

### 1. Install dependencies

```bash
pip install torch
```

### 2. Add training data

Drop a plain-text file at `input.txt` in the repo root. The classic choice is the [tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) corpus, but anything works — the tokenizer is built on the fly from the unique characters in the file.

### 3. Train

```bash
python train.py
```

Defaults (in `config.py`): a 4-layer, 4-head model with `d_model=64` and a context window of 64 characters, trained for 5,000 steps with AdamW at `lr=3e-4`. Eval loss prints every 500 steps. The final checkpoint lands at `pretrained/gpt.pt`.

CUDA is used automatically if available; otherwise it falls back to CPU.

### 4. Generate

```bash
python generate.py --prompt "ROMEO:" --max-new-tokens 500
```

Flags:

- `--prompt` — starting text (any characters not in the tokenizer's vocabulary are dropped).
- `--max-new-tokens` — how many tokens to generate after the prompt.
- `--checkpoint` — path to a `.pt` file (defaults to `config.checkpoint_path`).

## Configuration

All hyperparameters live in `config.py`. The defaults are sized for fast experimentation on a laptop, not quality:

| Parameter | Default | Notes |
|---|---|---|
| `d_model` | 64 | Embedding / hidden dimension |
| `num_heads` | 4 | Each head is `d_model / num_heads = 16` wide |
| `num_layers` | 4 | Transformer blocks |
| `chunk_size` | 64 | Maximum sequence length / context window |
| `batch_size` | 32 | |
| `max_steps` | 5000 | Training iterations |
| `lr` | 3e-4 | AdamW |
| `eval_interval` | 500 | Steps between train/val loss reports |

Bump `d_model`, `num_layers`, and `chunk_size` if you have a GPU and want a less-incoherent model.

## How attention is wired

Three things in `model.py` are worth singling out because they're the parts most implementations gloss over:

**Causal mask with a KV cache.** When a new token is fed in, `q` has length 1 but `k` covers the whole history. The mask is sliced as `tril[T_k - T_q : T_k, :T_k]` so row *i* of the mask corresponds to the absolute position of the *i*-th new query, and lets it attend to all keys up to and including itself.

**Position embeddings during cached generation.** `forward` takes a `start_pos` argument so the position embedding lookup uses absolute positions (`start_pos` to `start_pos + T`) rather than restarting at 0 for each new token.

**Cache invalidation at the boundary.** Once `cur_pos >= chunk_size`, there's no valid way to extend the cache (positions would exceed the embedding table), so `generate` recomputes from a truncated context. This makes long generations slower but always correct.

## Design notes / known limitations

- Attention is computed per-head in a Python loop rather than as a single batched matmul, which is slower than a "real" implementation but much easier to read.
- No dropout, weight tying, or learning-rate schedule — adding any of these is a small change.
- Tokenizer is character-level. Swap in BPE (e.g. `tiktoken`) if you want to train on real text efficiently.
- Sampling is temperature-only. Top-k and top-p are easy add-ons in `generate`.
- KV cache is rebuilt from scratch each time the context window fills, so very long generations lose the cache speedup.

## Reference

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) walkthrough was the starting point for the overall structure.
