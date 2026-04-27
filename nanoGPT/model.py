import torch


# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()


print(f"Total Chars: {len(text)}")
print(text[:100])