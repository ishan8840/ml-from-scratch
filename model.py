import torch
import torch.nn as nn
import torch.nn.functional as F


# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()


chars = ''.join(sorted(set(text)))
vocab_size = len(chars)
chunk_size = 8

# encoding and decoding
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}

encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(batch_size, chunk_size):
    x = torch.zeros([batch_size, chunk_size], dtype=torch.long)
    y = torch.zeros([batch_size, chunk_size], dtype=torch.long)

    for i in range(batch_size):
        idx = torch.randint(0, len(encoded) - chunk_size, (1,))
        xb = encoded[idx:idx+chunk_size]
        yb = encoded[idx+1:idx+chunk_size+1]
        x[i] = xb
        y[i] = yb
    
    return x, y


# Attention
class Head(nn.Module):

    def __init__(self, d_model, head_size, max_seq_len):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False) 
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.head_size = head_size
    
    def forward(self, x):
        T = x.shape[1]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = torch.softmax(wei * self.head_size**-0.5, dim=-1)
        attention = wei @ v

        return attention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, max_seq_len, num_heads):
        super().__init__()

        head_size = d_model // num_heads
        self.heads = nn.ModuleList([Head(d_model, head_size, max_seq_len) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model, bias=False)

    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    
    def __init__(self, d_model):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
        )
    
    def forward(self, x):
        return self.model(x)


class Block(nn.Module):

    def __init__(self, d_model, chunk_size, num_heads):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_seq_len=chunk_size)
        self.ffwd = FeedForward(d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)   # create once
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class GPT(nn.Module):

    def __init__(self, vocab_size, d_model, chunk_size, num_heads):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)  # (B, T, C)
        self.pos_encoding = nn.Embedding(chunk_size, d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, chunk_size, num_heads) for _ in range(6)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(idx.shape[1], device=idx.device)
        pos_emb = self.pos_encoding(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        B, T, V = logits.shape

        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

        return logits, loss
    

model = GPT(vocab_size, d_model=64, chunk_size=chunk_size)
xb, yb = get_batch(4, 8)
out = model(xb, chunk_size)

print(out.shape)