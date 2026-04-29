"""Transformer model definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, d_model, head_size, max_seq_len):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.head_size = head_size

    def forward(self, x, kv_cache=None):
        T = x.shape[1]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)  # (B, T_past + T_new, head_size)
            v = torch.cat([past_v, v], dim=1)

        new_cache = (k, v)

        T_q = q.shape[1]
        T_k = k.shape[1]

        wei = q @ k.transpose(-2, -1)  # (B, T_q, T_k)
        # rows correspond to positions [T_k - T_q, T_k); each row i can attend to keys [0, T_k - T_q + i]
        wei = wei.masked_fill(self.tril[T_k - T_q:T_k, :T_k] == 0, float('-inf'))
        wei = torch.softmax(wei * self.head_size ** -0.5, dim=-1)
        return wei @ v, new_cache, wei


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, max_seq_len, num_heads):
        super().__init__()
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([
            Head(d_model, head_size, max_seq_len) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        if kv_cache is None:
            kv_cache = [None] * len(self.heads)

        outs, new_caches, weis = [], [], []
        for h, c in zip(self.heads, kv_cache):
            o, nc, w = h(x, c)
            outs.append(o)
            new_caches.append(nc)
            weis.append(w)

        return self.proj(torch.cat(outs, dim=-1)), new_caches, weis


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.model(x)


class Block(nn.Module):
    def __init__(self, d_model, chunk_size, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_seq_len=chunk_size)
        self.ffwd = FeedForward(d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache, weis = self.mha(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))

        return x, new_cache, weis


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, chunk_size, num_heads, num_layers):
        super().__init__()
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(chunk_size, d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, chunk_size, num_heads) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        T = idx.shape[1]
        tok_emb = self.token_embedding(idx)
        # absolute positions of the *new* tokens
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        pos_emb = self.pos_encoding(pos)

        x = tok_emb + pos_emb

        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        new_caches, weights = [], []
        for block, c in zip(self.blocks, kv_caches):
            x, nc, weis = block(x, c)
            new_caches.append(nc)
            weights.append(weis)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None, new_caches, weights

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss, new_caches, weights

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        idx_cond = idx[:, -self.chunk_size:]
        logits, _, caches, _ = self(idx_cond, start_pos=0)

        # current absolute position of the next token to be generated
        cur_pos = idx_cond.shape[1]

        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

        for _ in range(max_new_tokens - 1):
            # if we'd exceed the context window, fall back to recomputing without cache
            if cur_pos >= self.chunk_size:
                idx_cond = idx[:, -self.chunk_size:]
                logits, _, caches, _ = self(idx_cond, start_pos=0)
                cur_pos = idx_cond.shape[1]
            else:
                logits, _, caches, _ = self(next_token, kv_caches=caches, start_pos=cur_pos)
                cur_pos += 1

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
    
    @torch.no_grad()
    def visualize_attention(self, idx, max_new_tokens):
        out = self.generate(idx, max_new_tokens)

        out_cond = out[:, -self.chunk_size:]
        _, _, _, weights = self(out_cond, start_pos=0)

        return out, weights