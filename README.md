# Preprint-Research-Paper-4-Data Poisoning in In-Context Learning: A Comparative Analysis on Tweet Sentiment


[![arXiv](https://img.shields.io/badge/arXiv-2510.03636-b31b1b.svg)](https://doi.org/10.48550/arXiv.2510.03636)  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

**Preprint (arXiv):** https://doi.org/10.48550/arXiv.2510.03636

---

## Authors & Contact

**Rabeya Amin Jhuma** — University of Information Technology and Sciences (UITS) — r.a.jhuma2019@gmail.com  
**Mostafa Mohaimen Akand Faisal** — University of Information Technology and Sciences (UITS) — mostafafaisal013@gmail.com

---


---

## Repository Overview

This repository contains all code, scripts, and analysis pipeline for the preprint:

**“FROM THEORY TO PRACTICE: EVALUATING DATA POISONING ATTACKS AND DEFENSES IN IN-CONTEXT LEARNING ON SOCIAL MEDIA HEALTH DISCOURSE”**

The work investigates how small adversarial perturbations (synonym replacement, negation insertion, randomized perturbation) in *support examples* can poison few-shot In-Context Learning (ICL) for sentiment analysis on HMPV tweets, and how a Spectral Signature Defense can mitigate those attacks.

---

## Table of Contents

- [Repository Structure](#repository-structure)  
- [Quick Start](#quick-start)  
- [Environment & Requirements](#environment--requirements)  
- [Pipeline (Text Flowchart)](#pipeline-text-flowchart)  
- [Scripts & Usage](#scripts--usage)  
- [Configuration](#configuration)  
- [Output & Results](#output--results)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Reproducibility](#reproducibility)  
- [Ethics & Data Privacy](#ethics--data-privacy)  
- [Citation](#citation)  
- [Authors & Contact](#authors--contact)  
- [License](#license)

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw_hmpv_tweets.csv
│   ├── cleaned_tweets.csv
│   ├── labeled_support.csv
│   └── targets.csv
├── scripts/
│   ├── 1_preprocess.py
│   ├── 2_generate_poison.py
│   ├── 3_build_prompts.py
│   ├── 4_icl_inference.py
│   ├── 5_evaluate.py
│   ├── 6_spectral_defense.py
│   └── 7_visualize.py
├── notebooks/
│   └── data-poisoning-hmpv.ipynb
├── results/
│   ├── metrics/
│   └── figures/
└── docs/
    └── methodology.md
```

> **Note:** Raw Twitter data is not included in the repository due to Twitter terms and privacy. See Data section below for collection instructions.

---

## Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Mostafa-Faisal/Preprint-Research-Paper-4-DataPoisoning-ICL-Health-Discourse.git
    cd Preprint-Research-Paper-4-DataPoisoning-ICL-Health-Discourse
    ```

2. **Install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate        # Linux / macOS
    # venv\Scripts\activate       # Windows
    pip install -r requirements.txt
    ```

3. **Prepare data and run minimal pipeline:**
    ```bash
    python scripts/1_preprocess.py --input data/raw_hmpv_tweets.csv --output data/cleaned_tweets.csv
    python scripts/2_generate_poison.py --support data/labeled_support.csv --ratio 0.25 --out data/support_poisoned_25.csv
    python scripts/3_build_prompts.py --support data/support_poisoned_25.csv --targets data/targets.csv --shots 5 --out results/prompts_25.json
    python scripts/4_icl_inference.py --prompts results/prompts_25.json --model zephyr --out results/icl_poisoned_25.csv
    python scripts/6_spectral_defense.py --support data/support_poisoned_25.csv --out data/support_cleaned_25.csv --flagged results/flagged_25.csv
    python scripts/5_evaluate.py --clean results/icl_clean_preds.csv --poisoned results/icl_poisoned_25.csv --out results/metrics_25.json
    ```

---

## Environment & Requirements

**Recommended:**
- Python 3.10+
- CPU is sufficient; GPU optional

**Libraries:**
```
pandas
numpy
scikit-learn
sentence-transformers
transformers
torch
nltk
spacy
textblob
matplotlib
seaborn
tqdm
pyyaml
```

Install with:
```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet punkt
python -m spacy download en_core_web_sm
```

---

## Pipeline (Text Flowchart)

```
Data Collection (raw_hmpv_tweets.csv)
        ↓
Data Preprocessing (cleaned_tweets.csv)
        ↓
Annotation & Split (labeled_support.csv, targets.csv)
        ↓
Poison Generation (synonym / negation / random)
        ↓
Support Sets (clean & poisoned: 25%, 50%, 75%, 100%)
        ↓
ICL Prompt Construction (NUM_SHOTS = 5)
        ↓
ICL Inference (Clean & Poisoned)
        ↓
Pre-Defense Evaluation (metrics + flip rate)
        ↓
Spectral Signature Defense (SVD → outlier flagging)
        ↓
Post-Defense Analysis (t-SNE, polarity, cluster)
        ↓
Visualization & Reporting
```


<details>
<summary><strong>Full Pipeline Diagram</strong> (click to expand)</summary>

```
                 ┌────────────────────────────┐
                 │  1. Data Collection         │
                 │  (raw_hmpv_tweets.csv)     │
                 └────────────┬───────────────┘
                              │
                 ┌────────────▼───────────────┐
                 │  2. Preprocessing          │
                 │  Cleaning, dedup, filter   │
                 └────────────┬───────────────┘
                              │
                 ┌────────────▼───────────────┐
                 │  3. Annotation & Split     │
                 │  Support vs Target sets    │
                 └────────────┬───────────────┘
                              │
                 ┌────────────▼───────────────┐
                 │  4. Poison Generation      │
                 │  (synonym / negation)      │
                 └────────────┬───────────────┘
                              │
                 ┌────────────▼───────────────┐
                 │  5. Support Construction   │
                 │  (Clean / 25–100% Poison)  │
                 └────────────┬───────────────┘
                              │
                 ┌────────────▼───────────────┐
                 │  6. ICL Prompt Formation   │
                 │  NUM_SHOTS = 5             │
                 └────────────┬───────────────┘
                              │
     ┌──────────────┬─────────┴─────────┬───────────────┐
     │              │                   │               │
┌────▼────┐    ┌────▼────┐        ┌────▼────┐      ┌────▼────┐
│ Clean   │    │ Poison  │        │ Defense │      │ Eval    │
│ ICL Run │    │ ICL Run │        │ (Spectral) │    │ Metrics │
└────┬────┘    └────┬────┘        └────┬────┘      └────┬────┘
     │              │                   │               │
     └──────────────┴───────────────────┴───────────────┘
                              │
                 ┌────────────▼───────────────┐
                 │  7. Post-Defense Analysis  │
                 │  t-SNE, Clusters, Polarity │
                 └────────────┬───────────────┘
                              │
                 ┌────────────▼───────────────┐
                 │  8. Visualization & Report │
                 │  (Figures, Tables)         │
                 └────────────────────────────┘
```
</details>

---

## Citation

If you use this work or code, please cite the preprint:

```
@article{ICL_Poisoning_2025,
  title={From Theory to Practice: Evaluating Data Poisoning Attacks and Defenses in In-Context Learning on Social Media Health Discourse},
  author={Rabeya Amin Jhuma and Mostafa Mohaimen Akand Faisal},
  journal={arXiv preprint arXiv:2510.03636},
  year={2025}
}
```


## License

This project is licensed under the **MIT License** — see `LICENSE` for details.

---

