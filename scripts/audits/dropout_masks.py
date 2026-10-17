#!/usr/bin/env python3
"""
dropout_masks.py — do two GPUs draw the same dropout masks, and compute the same numbers?
(journey §33.105)

Twins — one run repeated on the other machine — start from identical weights (initialised on the
CPU) and see the same batches; their first evaluations agree to 1e-9. Yet their training losses
differ from the first step. The model's dropout (0.1) draws its masks on the GPU from torch's CUDA
generator, and a kernel's threads take their random numbers by their place in its launch grid, which
depends on the GPU. This records, on the machine it runs on:

  masks       for the activation shapes of the base model, seeded as training seeds them, the
              SHA-256 of each dropout mask drawn on the GPU (and, for reference, on the CPU)
  arithmetic  a matrix product, a softmax and a layer norm on fixed inputs made on the CPU: the
              SHA-256 of each result, its float64 sum and its first values

Compare two machines' records: equal mask hashes mean equal masks; the arithmetic's values show
how far the GPUs' rounding differs where nothing is random.

    python scripts/audits/dropout_masks.py
"""
import os
import sys
import hashlib
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda', help='cpu only to test this script')
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from morphlm.record import AuditRecord
    from morphlm.config import ModelConfig, TrainConfig
    rec = AuditRecord('dropout_masks', a)
    m, tr = ModelConfig(), TrainConfig()
    B, L, D, H = tr.batch_size, m.seq_len if hasattr(m, 'seq_len') else 256, m.d_model, m.n_heads
    shapes = {'embedding (B, L, D)': (B, L, D), 'attention (B, H, L, L)': (B, H, L, L),
              'feed-forward (B, L, 4D)': (B, L, 4 * D)}
    dev = a.device
    gpu = dev.startswith('cuda')
    out = {'gpu': torch.cuda.get_device_name(0) if gpu else 'none (a test on the CPU)',
           'sm_count': torch.cuda.get_device_properties(0).multi_processor_count if gpu else None,
           'capability': list(torch.cuda.get_device_capability(0)) if gpu else None, 'masks': {}, 'arithmetic': {}}
    for name, shape in shapes.items():
        torch.manual_seed(a.seed)
        g = F.dropout(torch.ones(shape, device=dev), p=m.dropout, training=True) != 0
        torch.manual_seed(a.seed)
        c = F.dropout(torch.ones(shape), p=m.dropout, training=True) != 0
        out['masks'][name] = {'shape': list(shape), 'gpu_sha256': sha(g), 'kept_share_gpu': g.float().mean().item(),
                              'cpu_sha256': sha(c), 'gpu_equals_cpu': bool(torch.equal(g.cpu(), c))}
        print(f'  mask {name:26} kept {g.float().mean().item():.6f}  gpu {sha(g)[:16]}  cpu {sha(c)[:16]}')
    gen = torch.Generator().manual_seed(a.seed)
    x = torch.randn(B * L, D, generator=gen)
    w = torch.randn(D, 4 * D, generator=gen) / D ** 0.5
    s = torch.randn(B, H, L, L, generator=gen)
    for name, fn in (('matmul (B·L, D) × (D, 4D)', lambda d: x.to(d) @ w.to(d)),
                     ('softmax (B, H, L, L)', lambda d: torch.softmax(s.to(d), dim=-1)),
                     ('layer norm (B·L, D)', lambda d: F.layer_norm(x.to(d), (D,)))):
        yg, yc = fn(dev), fn('cpu')
        out['arithmetic'][name] = {'gpu_sha256': sha(yg), 'gpu_sum_f64': yg.double().sum().item(),
                                   'gpu_first': [float(v) for v in yg.flatten()[:6].cpu()],
                                   'largest_gpu_minus_cpu': (yg.cpu() - yc).abs().max().item()}
        print(f'  {name:26} gpu {sha(yg)[:16]}  sum {yg.double().sum().item():.12e}  |gpu−cpu| ≤ {(yg.cpu() - yc).abs().max().item():.2e}')
    rec.add('device', out)
    rec.write()


if __name__ == '__main__':
    main()
