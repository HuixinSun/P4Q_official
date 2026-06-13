# Checkpoints

CIFAR-100 / CLIP-ViT-B/32 / 4-4-8 ([Table 1](https://arxiv.org/abs/2409.17634)).

| Method | Top-1 | Top-5 | Size (MB) |
|--------|-------|-------|-----------|
| PTQ baseline (OMSE) | 46.05 | 74.47 | 94.76 |
| **P4Q** | **69.20** | **91.26** | 94.78 |

```
checkpoints/p4q/
├── p4q_prompt_learner.pth
└── p4q_adapter.pth
```

```bash
bash scripts/test_p4q.sh        # evaluate P4Q
bash scripts/test_baseline.sh   # PTQ baseline (no ckpt needed)
bash scripts/train_p4q.sh       # train from scratch
```
