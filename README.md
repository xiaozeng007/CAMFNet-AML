# CAMFNet-AML: Conflict-Aware Fusion Network with Adaptive Modal Leader for Multimodal Sentiment Analysis

## Anonymous Maintenance Statement
This repository is anonymously maintained for NeurIPS 2026 double-blind review.
- All personally identifiable information (authors, affiliations, emails, personal paths) has been removed.
- The code and documentation are provided only for peer review and reproducibility assessment.
- The full repository with author information will be made public upon acceptance.

## Abstract
Multimodal sentiment analysis often suffers from modality inconsistency, noisy cross-modal interactions, and unstable fusion behavior under domain shift or missing/weak modalities. We propose **CAMFNet-AML**, a conflict-aware fusion framework that explicitly models inter-modal conflicts and dynamically selects a reliable fusion leader at each step. CAMFNet-AML integrates (i) an Adaptive Modal Leader mechanism, (ii) multidimensional conflict-aware gating, and (iii) emotion triplet fusion with multi-head attention gating to improve robust multimodal representation learning. On the MUStARD++ dataset, CAMFNet-AML achieves state-of-the-art performance with an F1-score of **85.08% (mean ± std over 5 runs)**.

## Key Features / Contributions
1. **Adaptive Modal Leader (AML) Mechanism**  
   Dynamically selects the dominant modality signal for fusion guidance, instead of relying on a fixed modality priority.

2. **Multidimensional Conflict-Aware Gating**  
   Uses a five-dimensional conflict vector to characterize cross-modal disagreement and regulate fusion flow with conflict-sensitive gates.

3. **Emotion Triplet Fusion with Multi-Head Attention Gating**  
   Incorporates emotion-related triplet cues into fusion and refines interactions via multi-head attention gating for improved robustness and discriminability.

## Repository Structure
```text
neurips2026-cafnet-aml/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── run.py
├── cafnet_aml/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── transformers_compat.py
│   ├── config/
│   │   ├── config_regression.json
│   │   ├── config_tune.json
│   │   └── citations.json
│   ├── models/
│   │   ├── AMIO.py
│   │   ├── NewTask/
│   │   │   ├── CAMFN_conflict.py
│   │   │   └── ...
│   │   ├── modules/
│   │   │   ├── adaptive_leader.py
│   │   │   ├── conflict_perception.py
│   │   │   └── emotion_triplet.py
│   │   └── ...
│   ├── trains/
│   │   ├── ATIO.py
│   │   ├── NewTask/
│   │   │   └── CAMFN_trainer_conflict.py
│   │   └── ...
│   ├── utils/
│   └── datasets/
├── baselines/
│   ├── MISA/
│   ├── MULT/
│   ├── MuVaC/
│   ├── Self_MM/
│   ├── TETFN/
│   ├── TFN/
│   ├── ...
│   └── reproduction/
├── scripts/
│   ├── train_cafnet_aml.sh
│   ├── reproduce_table1_mustardpp.sh
│   ├── reproduce_mosei.sh
│   ├── run_ablation.sh
│   ├── reproduce_significance_mustardpp.sh
│   ├── run_significance_test.py
│   └── generate_paper_figures.py
├── configs/
├── data/
│   └── README.md
└── docs/
```

## Reproduction Instructions
This repository provides **helper scripts to assist reproduction of the main experimental results**.

> **Important reproducibility note**  
> Due to randomness in training (different seeds, hardware, PyTorch/CUDA versions, etc.), results may have small variance. The paper reports **mean ± std over 5 runs**.

### 1) Environment Setup
#### Option A: pip
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: conda
```bash
conda env create -f environment.yml
conda activate cafnet-aml
```

### 2) Prepare Data
1. Follow `data/README.md` to obtain and preprocess datasets/features.
2. Ensure processed feature files exist at the paths referenced in:
   - `cafnet_aml/config/config_regression.json`
   - `cafnet_aml/config/config_tune.json`
3. Keep dataset paths **relative** to the repository root when possible.

### 3) Run CAMFNet-AML Manually (Recommended Primary Path)
```bash
python run.py \
  -m camfnet_aml \
  -d mustardpp \
  -c cafnet_aml/config/config_regression.json \
  -s 1111 -s 1112 -s 1113 -s 1114 -s 1115 \
  --model-save-dir saved_models \
  --res-save-dir results \
  --log-dir logs \
  -n 0 \
  -v 1
```
Outputs are written to `results/` and logs to `logs/`.

### 4) Use Helper Scripts (Convenience)
- MUStARD++ main experiment:
```bash
bash scripts/reproduce_table1_mustardpp.sh
```
- CMU-MOSEI experiment:
```bash
bash scripts/reproduce_mosei.sh
```
- Ablation experiments:
```bash
bash scripts/run_ablation.sh
```
- Standard training shortcut:
```bash
bash scripts/train_cafnet_aml.sh
```

### 5) Statistical Significance Reproduction (MUStARD++)
Run paired significance tests (paired t-test, paired permutation test, Wilcoxon signed-rank test, Holm-Bonferroni correction, and bootstrap 95% CI):
```bash
bash scripts/reproduce_significance_mustardpp.sh
```
Expected output:
- `results_sig/statistics/mustardpp_camfn_vs_muvac_sig_10seeds.json`

### 6) Baseline Reproduction (Manual + Helper)
Example (manual):
```bash
python run.py -m mult -d mustardpp -c cafnet_aml/config/config_regression.json -s 1111 -s 1112 -s 1113
```
Example (helper):
```bash
bash baselines/reproduction/run_baseline_mustardpp.sh mult mustardpp
```

### FAQ
**Q1: Why are my numbers slightly different from the paper?**  
A: Small fluctuations are expected due to random seeds, GPU type, library versions, and low-level kernel nondeterminism. Compare against **mean ± std** trends rather than a single run.

**Q2: Which metric should I focus on for MUStARD++?**  
A: Follow the paper protocol and prioritize F1-based comparison on the official split, reporting mean ± std across 5 runs.

**Q3: Do scripts guarantee identical values to the paper table?**  
A: No absolute guarantees are provided. Scripts are intended to assist faithful protocol execution under the same settings.

**Q4: Why are raw datasets/checkpoints not included?**  
A: For anonymization, legal/data-license compliance, and repository size safety, raw/private artifacts are excluded.

## Datasets
This repository does **not** include raw dataset files.
- Please follow `data/README.md` for dataset access and preprocessing instructions.
- Supported benchmarks in this codebase include MUStARD++ and CMU-MOSEI (and additional datasets configured in JSON files).
- Ensure your local processed features match expected schema and split definitions in config files.

## Baselines
All comparison baselines used in the paper are organized under `baselines/`, including representative models such as:
- MISA
- MULT
- MuVaC
- Self-MM
- TETFN
- TFN
- and other baseline families used in the experimental matrix

Each baseline directory contains corresponding implementation files, and `baselines/reproduction/` provides helper scripts for baseline execution.

## Citation
If you find this repository useful, please cite the following anonymized submission during review:
```bibtex
@inproceedings{anonymous2026cafnetaml,
  title={CAMFNet-AML: Conflict-Aware Fusion Network with Adaptive Modal Leader for Multimodal Sentiment Analysis},
  author={Anonymous Authors},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License
This project is released under the **MIT License**. See `LICENSE` for details.
