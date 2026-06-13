# Checkpoints

Pretrained weights that reproduce the **P4Q** row in Table 1 of the paper (CIFAR-100, CLIP ViT-B/32, 4-4-8).

| Method | Top-1 | Top-5 | Size (MB) |
|--------|-------|-------|-----------|
| PTQ baseline (OMSE) | 46.05 | 74.47 | 94.76 |
| **P4Q** | **69.20** | **91.26** | 94.78 |

| File | Component |
|------|-----------|
| `p4q_prompt_learner.pth` | Learnable prompt |
| `p4q_adapter.pth` | Low-bit adapter |

```
checkpoints/p4q/
├── p4q_prompt_learner.pth
└── p4q_adapter.pth
```

```bash
bash scripts/test_p4q.sh        # evaluate with released checkpoints
bash scripts/test_baseline.sh   # PTQ baseline (no checkpoint required)
bash scripts/train_p4q.sh       # train from scratch
```
