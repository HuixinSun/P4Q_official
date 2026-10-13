<h1 align="center">P4Q: Learning to Prompt for Quantization in Low-Bit CLIP</h1>

<p align="center" style="font-size: 1.05em">
  <strong>Huixin Sun, Runqi Wang, Yanjing Li, Chang Gao, Liping Jing, Xiaolong Jiang, Yao Hu, Baochang Zhang, Xianbin Cao</strong><br/>
  <a href="https://dl.acm.org/doi/10.1145/3822594"><img src="https://img.shields.io/badge/ACM_TOMM-2026-2563eb?style=flat-square" alt="ACM TOMM 2026"/></a>
  <a href="https://dl.acm.org/doi/10.1145/3822594"><img src="https://img.shields.io/badge/Paper-ACM-b31b1b?style=flat-square" alt="Paper"/></a>
  <a href="models/"><img src="https://img.shields.io/badge/Code-P4Q-16a34a?style=flat-square" alt="Code"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-9333ea?style=flat-square" alt="License"/></a>
</p>

Official PyTorch implementation of [**P4Q: Learning to Prompt for Quantization in Low-Bit CLIP**](https://dl.acm.org/doi/10.1145/3822594).

P4Q integrates Post-Training Quantization (PTQ) with Parameter-Efficient Fine-Tuning (PEFT) to reduce computational costs and bridge domain gaps. It introduces learnable prompts to embed downstream knowledge, low-bit adapters to realign image and text feature distributions, and a distillation loss based on cosine similarity predictions to transfer knowledge from a full-precision teacher model.

<p align="center">
  <img src="figures/framework.png" width="880"/>
  <br/>
  <em>Overview of P4Q. The blue and green parts represent the visual and textual streams, respectively, and the trainable parameters are marked by the fire icon. QFC denotes a quantized fully connected layer. (a) Quantized CLIP. (b) Knowledge distillation.</em>
</p>

## Released Code and Results

- P4Q and CLIP PTQ implementation
- Training and evaluation scripts
- Pretrained [checkpoints](checkpoints/p4q/) for CIFAR-100 with CLIP ViT-B/32 (4-4-8)

Results on CIFAR-100 with CLIP ViT-B/32 under the 4-4-8 setting:

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| FP32 CLIP (zero-shot) | 65.34 | 89.00 |
| PTQ baseline (OMSE) | 46.05 | 74.47 |
| **P4Q** | **69.20** | **91.26** |

## Reproduction

**Environment.**

```bash
conda create -n p4q python=3.10 -y && conda activate p4q
pip install -r requirements.txt
```

**Dataset.** Download the CIFAR-100 Python dataset and extract it to:

```text
data/cifar-100-python/
├── meta
├── train
└── test
```

Use `DATA_ROOT=/path/to/data` if the dataset is stored elsewhere.

**Evaluation.** Run the PTQ baseline under the 4-4-8 setting:

```bash
bash scripts/test_baseline.sh
```

This reproduces **46.05 Top-1** and **74.47 Top-5**. To evaluate P4Q, the following command loads `p4q_prompt_learner.pth` and `p4q_adapter.pth` from `checkpoints/p4q/`:

```bash
bash scripts/test_p4q.sh
```

This reproduces **69.20 Top-1** and **91.26 Top-5**.

**Training.** Train P4Q from scratch, then evaluate the resulting checkpoints:

```bash
SAVE_DIR=checkpoints/p4q_retrained bash scripts/train_p4q.sh
CKPT_DIR=checkpoints/p4q_retrained bash scripts/test_p4q.sh
```

Use `CUDA_VISIBLE_DEVICES=<GPU_ID>` to select a GPU.

## Citation

```bibtex
@article{sun2026p4q,
  title     = {P4Q: Learning to Prompt for Quantization in Low-Bit CLIP},
  author    = {Sun, Huixin and Wang, Runqi and Li, Yanjing and Gao, Chang and Jing, Liping and Jiang, Xiaolong and Hu, Yao and Zhang, Baochang and Cao, Xianbin},
  journal   = {ACM Transactions on Multimedia Computing, Communications, and Applications},
  publisher = {Association for Computing Machinery},
  year      = {2026},
  doi       = {10.1145/3822594}
}
```

## License

See [LICENSE](LICENSE).
