# XGaze: Cross-Attention based Gaze Estimation

Learnable query-based cross-attention for gaze target detection, extended to multi-person
supervision (in/out-of-frame prediction) and evaluation on **GazeFollow**,
**VideoAttentionTarget (VAT)**, and **ChildPlay**.

> Built on top of [Toward Semantic Gaze Target Detection (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/dbeb7e621d4a554069a6a775da0f7273-Paper-Conference.pdf) by Tafasca et al. See [Citation](#citation).

> ⚠️ This is an early draft README.

---

## 1. Environment Setup

```shell
git clone https://github.com/JinWoong-Jung/XGaze.git
cd XGaze

conda create -n XGaze python=3.11.0
conda activate XGaze
pip install -r requirements.txt
```

Pre-trained initialization weights (Gaze360 ResNet18, DINOv3 ViT-B) are expected under
`weights/`. Trained checkpoints go under `checkpoints/`.

---

## 2. Dataset Setup

Download each dataset, then point the config files to your local paths.

| Dataset | Used for | Config file |
|---|---|---|
| GazeFollow | training + test | `XGaze/conf/config_gf.yaml` |
| VideoAttentionTarget | fine-tuning / zero-shot test | `XGaze/conf/config_vat.yaml` |
| ChildPlay | fine-tuning / zero-shot test | `XGaze/conf/config_childplay.yaml` |

In each config, set the dataset root(s), e.g.:

```yaml
data:
    gf:
        root: /path/to/gazefollow_extended
    vat:
        root: /path/to/VideoAttentionTarget
    childplay:
        root: /path/to/ChildPlay
```

Also update `project.root` and `model.XGaze.hf_image_encoder_local_dir` to your machine's paths.

---

## 3. Training & Testing

The entry point is `main.py` (Hydra + PyTorch Lightning). Select the dataset via `--config-name`.

**Train on GazeFollow (from scratch), then test:**
```shell
python main.py --config-name=config_gf experiment.task="train+test"
```

**VAT / ChildPlay** — warm-start from a GazeFollow checkpoint, then fine-tune or run zero-shot:
```shell
# zero-shot test only
python main.py --config-name=config_vat experiment.task="test" \
    model.weights="checkpoints/best_gf.ckpt"

# fine-tune then test
python main.py --config-name=config_vat experiment.task="train+test" \
    model.weights="checkpoints/best_gf.ckpt"
```
(Replace `config_vat` with `config_childplay` for ChildPlay.)

SLURM submission scripts are provided: `train_gf.sh`, `train_vat.sh`, `train_cp.sh`.

---

## 4. Results

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="3" align="center">GazeFollow</th>
      <th colspan="3" align="center">VideoAttentionTarget</th>
      <th colspan="3" align="center">ChildPlay</th>
    </tr>
    <tr>
      <th align="center">AUC &uarr;</th>
      <th align="center">Avg L2 &darr;</th>
      <th align="center">Min L2 &darr;</th>
      <th align="center">AUC &uarr;</th>
      <th align="center">L2 &darr;</th>
      <th align="center">AP<sub>in/out</sub> &uarr;</th>
      <th align="center">AUC &uarr;</th>
      <th align="center">L2 &darr;</th>
      <th align="center">AP &uarr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Gaze-LLE (ViT-B)</td>
      <td align="center">0.956</td><td align="center">0.104</td><td align="center">0.045</td>
      <td align="center">0.933</td><td align="center">0.107</td><td align="center">0.897</td>
      <td align="center">0.949</td><td align="center">0.106</td><td align="center">0.994</td>
    </tr>
    <tr>
      <td align="left">Gaze-LLE (ViT-L)</td>
      <td align="center">0.958</td><td align="center">0.099</td><td align="center">0.041</td>
      <td align="center">0.937</td><td align="center">0.103</td><td align="center">0.903</td>
      <td align="center">0.951</td><td align="center">0.101</td><td align="center">0.994</td>
    </tr>
    <tr>
      <td align="left"><b>Ours*</b></td>
      <td align="center">0.950</td><td align="center">0.098</td><td align="center">0.044</td>
      <td align="center">0.969</td><td align="center">0.095</td><td align="center">0.658</td>
      <td align="center">0.975</td><td align="center">0.093</td><td align="center">0.972</td>
    </tr>
  </tbody>
</table>

\* **Ours** — VideoAttentionTarget and ChildPlay results are **zero-shot** (no fine-tuning);
GazeFollow is the trained result. See [`results.md`](./results.md) for details.

---

## Citation

```bibtex
@article{tafasca2024toward,
  title={Toward Semantic Gaze Target Detection},
  author={Tafasca, Samy and Gupta, Anshul and Bros, Victor and Odobez, Jean-Marc},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={121422--121448},
  year={2024}
}
```

## Acknowledgement
This codebase builds on [Sharingan](https://github.com/idiap/sharingan) /
[MultiMAE](https://github.com/EPFL-VILAB/MultiMAE). We thank the authors for their contributions.
