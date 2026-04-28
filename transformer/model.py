import torch
import torch.nn as nn


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



class GPT(nn.Module):

    def __init__(self, vocab_size, d_model, chunk_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # (B, T, C)
        self.pos_encoding = nn.Embedding(chunk_size, d_model)
    
    def forward(self, idx, chunk_size):
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(idx.shape[1])
        pos_emb = self.pos_encoding(pos)
        x = tok_emb + pos_emb

        return x
    
model = GPT(vocab_size, d_model=64, chunk_size=chunk_size)
xb, yb = get_batch(4, 8)
out = model(xb, chunk_size)

print(out.shape)