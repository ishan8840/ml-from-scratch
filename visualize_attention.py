"""Load a trained checkpoint and visualize attention weights.

Usage:
    python visualize_attention.py --prompt "Hello" --max-new-tokens 50
    python visualize_attention.py --layer 2 --head 1
"""
import argparse
import torch
import matplotlib.pyplot as plt

from config import Config
from model import GPT


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt['config']
    itos = ckpt['itos']
    stoi = ckpt['stoi']

    model = GPT(
        vocab_size=len(itos),
        d_model=cfg_dict['d_model'],
        chunk_size=cfg_dict['chunk_size'],
        num_heads=cfg_dict['num_heads'],
        num_layers=cfg_dict['num_layers'],
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, stoi, itos


def labels_from_ids(ids, itos):
    """Turn token ids into display strings (escape newlines so axes don't break)."""
    out = []
    for i in ids:
        c = itos[i]
        if c == '\n':
            out.append('\\n')
        elif c == ' ':
            out.append('␣')
        else:
            out.append(c)
    return out


def plot_single_head(weights, labels, layer, head, save_path):
    attn = weights[layer][head][0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Key (attended to)')
    ax.set_ylabel('Query (attending from)')
    ax.set_title(f'Layer {layer}, Head {head}')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f'Saved: {save_path}')


def plot_layer_grid(weights, labels, layer, save_path):
    layer_w = weights[layer]
    n_heads = len(layer_w)
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for h, ax in enumerate(axes.flat):
        if h >= n_heads:
            ax.axis('off'); continue
        attn = layer_w[h][0].cpu().numpy()
        ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_title(f'Head {h}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Layer {layer} — all heads', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f'Saved: {save_path}')


def plot_all_layers_avg(weights, labels, save_path):
    n_layers = len(weights)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for l, ax in enumerate(axes.flat):
        if l >= n_layers:
            ax.axis('off'); continue
        stacked = torch.stack([h[0] for h in weights[l]], dim=0)  # (H, T, T)
        avg = stacked.mean(dim=0).cpu().numpy()
        ax.imshow(avg, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {l} (avg over heads)', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Average attention per layer', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f'Saved: {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='',
                        help='Optional starting text. Defaults to a single newline.')
    parser.add_argument('--max-new-tokens', type=int, default=50)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint. Defaults to Config.checkpoint_path.')
    parser.add_argument('--layer', type=int, default=0,
                        help='Which layer to show in the detail/grid plots.')
    parser.add_argument('--head', type=int, default=0,
                        help='Which head to show in the detail plot.')
    parser.add_argument('--show', action='store_true',
                        help='Open plots in a window in addition to saving.')
    args = parser.parse_args()

    cfg = Config()
    ckpt_path = args.checkpoint or cfg.checkpoint_path

    model, stoi, itos = load_model(ckpt_path, cfg.device)

    if args.prompt:
        ids = [stoi[c] for c in args.prompt if c in stoi]
        if not ids:
            ids = [0]
    else:
        ids = [0]

    context = torch.tensor([ids], dtype=torch.long, device=cfg.device)

    # generate, then one clean forward pass for square attention matrices
    out, weights = model.visualize_attention(context, max_new_tokens=args.max_new_tokens)

    # decode and print the generated text
    decoded = ''.join(itos[i] for i in out[0].tolist())
    print('Generated:')
    print(decoded)
    print()
    print(f'Attention: {len(weights)} layers x {len(weights[0])} heads '
          f'x {tuple(weights[0][0].shape)}')

    # the visualized window is the last chunk_size tokens of the output
    visualized_ids = out[0, -model.chunk_size:].tolist()
    labels = labels_from_ids(visualized_ids, itos)

    plot_all_layers_avg(weights, labels, 'attn_layers_avg.png')
    plot_layer_grid(weights, labels, args.layer, f'attn_layer{args.layer}_heads.png')
    plot_single_head(weights, labels, args.layer, args.head,
                     f'attn_layer{args.layer}_head{args.head}.png')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()