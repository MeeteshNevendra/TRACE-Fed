# TRACE-Fed

TRACE-Fed: Communication-Efficient, Adapter-Based Federated Learning with Trust-Weighted Aggregation for Brain Tumour MRI.

In TRACE-Fed, the clients of a federation keep an ImageNet-pretrained ConvNeXt-T backbone frozen. They train only low-rank (LoRA) adapters and the layers that TRACE-Net adds on top of the backbone, and exchange only these parameters. The updates are compressed in both directions (top-k sparsification and 8-bit quantisation with error feedback), and the server aggregates them with a discount-only trust rule.

The code covers the whole study:
- the duplicate audit of the two public datasets and the deduplicated splits;
- centralised training of TRACE-Net and six ImageNet-pretrained baselines (ResNet-18, ResNet-50, EfficientNet-B0, ConvNeXt-T, ViT-S and Swin-T);
- federated training of TRACE-Fed and eight baselines (FedAvg, FedProx, SCAFFOLD, FedAdam, FedBN, FedPer, FedRep and MOON);
- validation-only model selection, ablations, heterogeneity and robustness experiments;
- statistics, Grad-CAM evaluation, and the manuscript's tables and figures.

This repository contains code only: no datasets, run records or model checkpoints.

## Contents

| Path | Content |
|---|---|
| `trace_fed/models/` | TRACE-Net (`trace_net.py`, with the selective state-space mixer `ssm.py` and the LoRA adapters `lora.py`) and the timm baselines (`baselines.py`) |
| `trace_fed/tools/check_duplicates.py` | Exact and near-duplicate audit across both datasets |
| `trace_fed/data.py` | Deduplicated catalogue, splits, image cache and on-GPU augmentation |
| `trace_fed/engine.py` | Training, evaluation and metrics |
| `trace_fed/fl.py` | Federated simulation: TRACE-Fed and the eight baselines |
| `trace_fed/run_central.py`, `trace_fed/run_fl.py` | One centralised or federated experiment |
| `trace_fed/pipeline.py` | The full study, stage by stage |
| `trace_fed/report.py` | Tables, statistics and figures from the run results |
| `trace_fed/xai.py` | Grad-CAM evaluation against the tumour boxes |
| `trace_fed/paper_assets.py` | The manuscript's numbers, tables and figures |
| `trace_fed/registry.py` | Model specification strings |

## Requirements

- **Hardware:** Python 3.14 and a CUDA GPU. The experiments ran with Python 3.14.5 on one NVIDIA RTX 4000 Ada (20 GB).
- **PyTorch:** install the CUDA build for your GPU first, from https://pytorch.org/get-started/locally/. The experiments used PyTorch 2.12.1 with CUDA 13.0.
- **Other packages:** `pip install -r requirements.txt` installs the versions the experiments used.
- **Pretrained weights:** timm downloads them from the Hugging Face Hub on first use.

## Data

The datasets are not included. Download these two Kaggle releases:

| | D1 | D2 |
|---|---|---|
| Title | MRI for Brain Tumor with Bounding Boxes | Brain MRI Images |
| Kaggle dataset | `ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes`, version 1 | `mhantor/mri-based-brain-tumor-images`, version 2 |
| Licence declared by the release | CC0: Public Domain | CC BY-NC-ND 4.0 |
| `archive.zip` | 139,133,481 bytes | 18,866,250 bytes |
| SHA-256 | `48eb5c6fa58f5e452caa8a95858a4ac0d3c0edbf7a3adb326c515a14f686f4ee` | `9e3aa7431afc7ea53a4411e16ec0d6613f29402e4da9b25e17f7812fafb3d0b5` |

Use the Download button on each dataset's Kaggle page. Alternatively, Kaggle's download endpoint:

```
curl -L -o d1_archive.zip "https://www.kaggle.com/api/v1/datasets/download/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes?datasetVersionNumber=1"
curl -L -o d2_archive.zip "https://www.kaggle.com/api/v1/datasets/download/mhantor/mri-based-brain-tumor-images?datasetVersionNumber=2"
```

Check each archive's SHA-256 against the table, with `sha256sum` or, in PowerShell, `Get-FileHash`. A different value means that Kaggle now serves different files.

Extract D1 into `archive/` and D2 into `archive_2/` at the repository root:

```
archive/Train/<Glioma|Meningioma|No Tumor|Pituitary>/{images,labels}/...
archive/Val/<Glioma|Meningioma|No Tumor|Pituitary>/{images,labels}/...
archive_2/Brain MRI Images/Brain MRI Images/<Train|Validation>/<Normal|Tumor>/...
```

## Running

Run every command from the repository root. All output goes to `results_v2/`.

1. **Duplicate audit,** required before any training:
   ```
   python -m trace_fed.tools.check_duplicates
   ```
   It writes `results_v2/data_audit/duplicate_groups.csv`. The first training run then collapses each duplicate group to one image, draws the splits and saves them to `results_v2/splits/splits.json`:
   - D1: a stratified 70/15/15 split into training, validation and test;
   - D2: stratified five-fold cross-validation.
2. **The full study:**
   ```
   python -m trace_fed.pipeline --stages all
   ```
   The stages run in this order:
   - `search_central` and `search_fl`: validation-only searches, with seeds 100 and 101 (centralised) and seed 100 (federated);
   - `select`: chooses the configurations from validation scores only and writes `results_v2/selection.json`;
   - `central` and `fl_main`: the final comparisons, with seeds 0-4 on D1 and the five folds on D2;
   - `ablation`, `hetero`, `robust`, `mixed` and `cross`: component ablations, non-IID and client-count sweeps, a corrupted client, a mixed-site federation and cross-dataset generalisation.

   Every stage skips runs whose result file already exists, so an interrupted pipeline resumes where it stopped. To run only some stages, list them, for example `--stages search_central search_fl select`.
3. **Single experiments,** for example:
   ```
   python -m trace_fed.run_central --dataset dataset1 --model trace --seeds 0 1 2 3 4
   python -m trace_fed.run_fl --dataset dataset1 --method fedavg fedprox trace --seeds 0 1 2
   python -m trace_fed.run_fl --dataset dataset1 --method trace --opts "trust=0" --tag no_trust
   ```
4. **Analysis:**
   ```
   python -m trace_fed.report        # tables, statistics and figures -> results_v2/tables, results_v2/figures
   python -m trace_fed.xai           # Grad-CAM evaluation of the saved centralised checkpoints
   python -m trace_fed.paper_assets  # the manuscript's numbers, tables and figures -> paper_v2/
   ```

**Reproducibility.** By default, GPU training is not bit-for-bit repeatable, so repeated runs can differ slightly. Two `run_fl.py` options help; both are off by default.
- **`--deterministic`** allows only deterministic GPU operations, and the run stops if an operation has none. It needs `PYTHONHASHSEED` to be set before Python starts (in PowerShell: `$env:PYTHONHASHSEED=0`):
  ```
  PYTHONHASHSEED=0 python -m trace_fed.run_fl --dataset dataset2 --method trace --seeds 2 --deterministic --provenance
  ```
- **`--provenance`** adds to the result file the settings, the software and GPU environment (including the processes using the GPU at the start), and the data and model hashes.

## Licence

The code is released under the MIT License (see `LICENSE`).

The datasets are not part of this repository. Each remains under the licence its release declares. D1's images are byte-identical to files of earlier Kaggle collections, which declare other licences.

## Citation

Citation details will be added when the paper is published.
