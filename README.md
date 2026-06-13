# P4Q: Learning to Prompt for Quantization in Low-Bit CLIP

[![arXiv](https://img.shields.io/badge/arXiv-2409.17634-b31b1b.svg)](https://arxiv.org/abs/2409.17634)

Official PyTorch implementation of [**P4Q**](https://arxiv.org/abs/2409.17634).

**P4Q** learns lightweight, quantization-aware prompts and adapters to restore image and text alignment under low-bit PTQ, while keeping the CLIP backbone frozen. On CIFAR-100 with ViT-B/32 at 4-4-8 bit setting, P4Q achieves **69.20%** Top-1 accuracy, outperforming both the PTQ baseline (46.05%) and FP32 zero-shot CLIP (65.34%).

> **P4Q: Learning to Prompt for Quantization in Low-Bit CLIP**  
> Huixin Sun, Runqi Wang, Yanjing Li, Chang Gao, Liping Jing, Xiaolong Jiang, Yao Hu, Baochang Zhang, Xianbin Cao  
> [arXiv:2409.17634](https://arxiv.org/abs/2409.17634)

<p align="center">
  <img src="figures/framework.png" width="880"/>
  <br/>
  <em>P4Q framework. (a) Quantized CLIP. (b) Teacher and student distillation.</em>
</p>

## Why P4Q?

- **Restores multimodal alignment** under aggressive low-bit quantization (e.g., 4-bit weight and activation)
- **Outperforms FP32 zero-shot CLIP** with only a small prompt and adapter overhead
- **Plug and play**: built on standard PTQ calibration and compatible with diverse observers and quantizers

This repository provides a modular CLIP PTQ toolkit, training and evaluation scripts, pretrained checkpoints, and a reproducible PTQ baseline.

## CLIP PTQ Toolkit

A modular quantization toolkit based on [FQ-ViT](https://github.com/linyang-zhh/FQ-ViT) style operators in `models/ptq/`, supporting flexible calibration recipes and bit width configurations:

| Module | Support |
|--------|---------|
| **Observers** | MinMax, EMA, OMSE, Percentile, PTF |
| **Quantizers** | Uniform, Log2 (LIS softmax) |
| **Q-layers** | `QLinear`, `QConv2d`, `QAct`, `QMultiheadAttention`, `QIntLayerNorm`, `QIntSoftmax` |
| **Bit-width** | `--bit_type {8,4,3,2}` for W/A; attention at 8-bit |

Default recipe for CIFAR-100 with 4-4-8 bit setting: MinMax per-channel weight quantization, OMSE per-channel activation quantization, and 8-bit attention.

```bash
python main.py --model ViT-B/32 --db_name cifar100 --root ./data \
    --quant --bit_type 4 --quant-method omse \
    --calib-iter 10 --zeroshot_prompt --val_quant --num-workers 0
```

## Released Results (CIFAR-100, ViT-B/32, 4-4-8)

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| FP32 CLIP (zero-shot) | 65.34 | 89.00 |
| PTQ baseline (OMSE) | 46.05 | 74.47 |
| **P4Q** | **69.20** | **91.26** |

Checkpoints are available at [`checkpoints/p4q/`](checkpoints/p4q/).

## Quick Start

```bash
conda create -n p4q python=3.10 -y && conda activate p4q
pip install -r requirements.txt

# Place CIFAR-100 under data/cifar-100-python/{meta,train,test}

bash scripts/test_baseline.sh   # PTQ baseline
bash scripts/test_p4q.sh        # P4Q evaluation
bash scripts/train_p4q.sh       # train P4Q
```

## Citation

```bibtex
@article{sun2024p4q,
  title   = {P4Q: Learning to Prompt for Quantization in Visual-language Models},
  author  = {Sun, Huixin and Wang, Runqi and Li, Yanjing and Cao, Xianbin and Jiang, Xiaolong and Hu, Yao and Zhang, Baochang},
  journal = {arXiv preprint arXiv:2409.17634},
  year    = {2024}
}
```

## Acknowledgements

[CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp), [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter), and [FQ-ViT](https://github.com/linyang-zhh/FQ-ViT).

## License

See [LICENSE](LICENSE).
