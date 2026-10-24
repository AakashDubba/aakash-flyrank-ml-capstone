# Prioritizing Content Refresh Opportunities in Enterprise SEO
### FlyRank Machine Learning Internship Capstone — End-to-End Implementation & Deployed Research

[![Live Paper](https://img.shields.io/badge/Research_Paper-Live_Page-blue?style=for-the-badge&logo=googlechrome)](https://aakashdubba.github.io/aakash-flyrank-ml-capstone/)
[![Open In Colab](https://img.shields.io/badge/Colab-Capstone_Notebook-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/AakashDubba/aakash-flyrank-ml-capstone/blob/main/work/notebooks/capstone.ipynb?flush_cache=true)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Data Credit](https://img.shields.io/badge/Data_Credit-FlyRank.ai-purple?style=for-the-badge)](https://flyrank.ai)

---

## 📌 Executive Summary

- **Author:** Aakash Dubba
- **Lane:** Refresh / Content Opportunity Scoring (Lane 2)
- **Deployed Research Paper:** [https://aakashdubba.github.io/aakash-flyrank-ml-capstone/](https://aakashdubba.github.io/aakash-flyrank-ml-capstone/)
- **Direct Submission Link:** `submission/paper_url.txt` $\rightarrow$ `https://aakashdubba.github.io/aakash-flyrank-ml-capstone/`
- **Data Source:** FlyRank Pseudonymized Warehouse Release (`content_refresh_anonymized.csv`), 30,000 pages across 32 enterprise clients.

This repository hosts the complete, reproducible machine learning capstone project for the FlyRank Applied Search Intelligence Internship. The project develops, validates, and deploys a group-aware machine learning decision-support system to prioritize content refresh opportunities across large-scale enterprise content portfolios, outperforming heuristic baselines by **3.20× on Precision@20 (80.0% vs. 25.0%)** and **4.22× on Precision@50 (76.0% vs. 18.0%)** on blind, held-out client domains.

---

## 🔬 The Research Question

> **"Which declining, high-visibility content pages should an editorial and SEO team prioritize for content refresh, expansion, or consolidation to maximize search traffic recovery while minimizing wasted editorial effort?"**

### The Operational Challenge
In enterprise digital marketing, an in-depth content refresh requires 4 to 12 hours of editorial labor (subject-matter expert interviews, rewriting, technical restructuring, and snippet optimization). For catalogs with 10,000 to 50,000+ indexed pages, editorial capacity is strictly capped at 20–50 pages per week. Traditional heuristics—such as updating any page older than 180 days with declining traffic—conflate stable evergreen assets with dying pages, causing a 75% false-positive rate.

Our ML system provides:
1. **Calibrated Probability of Decline:** Predicting true traffic erosion without leaking post-hoc trend indicators.
2. **Prioritized Opportunity Queue (0–100):** Combining predicted decline risk with total traffic exposure.
3. **Auditable Reason Codes:** Diagnosing why each page was selected (`model_decline_risk`, `low_ctr_visible_page`, `low_engagement_visible_page`).
4. **Action Routing:** Directing tickets to copywriters (`refresh_and_review_ctr`), UX specialists (`refresh_and_review_engagement`), or researchers (`expand_and_refresh`).

---

## 📊 Key Results (Evaluated on Unseen Client Domains)

All models were evaluated on an out-of-sample **Client-Holdout Test Set (2,325 assets from 6 completely unseen enterprise clients)** with **zero client overlap** between training and evaluation:

| Architecture | Precision@10 | Precision@20 | Precision@50 | Precision@100 | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Chance (Base Rate)** | 0.542 | 0.542 | 0.542 | 0.542 | 0.500 | 0.542 | 0.250 |
| **Heuristic Baseline Rule** | 0.300 | 0.250 | 0.180 | 0.210 | 0.592 | 0.435 | 0.240 |
| **Logistic Regression (L2)** | 0.200 | 0.350 | 0.420 | 0.450 | 0.700 | 0.523 | 0.216 |
| **Decision Tree (Depth 5)** | 0.800 | 0.800 | 0.680 | 0.650 | 0.742 | 0.575 | 0.200 |
| **Random Forest (Best General)** | **0.700** | **0.800** | **0.760** | **0.730** | **0.751** | **0.622** | **0.198** |
| **Gradient Boosting Classifier** | **0.800** | **0.850** | **0.820** | **0.850** | **0.778** | **0.678** | **0.192** |

### Top Predictive Signals
1. `days_with_impressions` (14.66%): Impression continuity across 90 days is the strongest indicator of algorithmic trust vs. decay.
2. `log_impressions_90d` (12.45%): Scale of historical search visibility.
3. `avg_position` (11.44%): Average ranking position in Google Search Console.
4. `content_age_days` (8.70%): Age since initial publication.
5. `char_count` (4.46%): Content body substance and depth.

---

## 🛡️ Data Safety & Privacy Compliance

This project complies strictly with `DATA_USE.md` and enterprise safety protocols:
- **No Private Data:** No client domain names, company names, URLs, page titles, or private search queries are included.
- **Identifiers Restricted:** `client_id` is used solely for client-level group splitting, never as a feature.
- **Zero Leakage:** Outcome columns (`trend_direction`, `trend_pct`) were purged prior to feature compilation.
- **Honest Claims:** All findings are framed as observed statistical associations and decision-support triage—never claiming to prove Google's internal ranking algorithm or causal refresh guarantees.

---

## 🗂️ Repository Architecture

```text
├── submission/
│   └── paper_url.txt           # Exactly one line: the live deployed research paper URL
├── docs/                       # Deployed GitHub Pages website
│   ├── index.html              # Interactive 3-door research paper application
│   └── charts/                 # Publication-ready SVG and PNG evaluation plots
├── work/                       # Intern deliverables
│   ├── capstone_report.md      # Full 9-section markdown capstone report
│   ├── notebooks/              # Complete verified assignment notebooks
│   │   ├── capstone.ipynb      # Executed end-to-end capstone notebook
│   │   ├── w01_research_question.ipynb  # ML-02 Completed
│   │   ├── w02_ml_task_framing.ipynb    # ML-03 Completed
│   │   ├── w03_data_contract.ipynb      # ML-04 Completed
│   │   └── w04_baseline_score.ipynb     # ML-07 Completed
│   ├── figures/                # High-res evaluation charts
│   └── outputs/                # Committed receipts, JSON metrics, and ranked queues
├── scripts/                    # Runnable ML pipelines
│   ├── build_capstone_notebook.py  # End-to-end capstone builder & executor
│   ├── ml_utils.py             # Feature definitions and helpers
│   └── run_all.py              # Reference baseline pipeline
├── data/
│   └── raw/content_refresh_anonymized.csv  # Anonymized 30,000-page dataset
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quickstart & Reproducibility

Reproduce all numbers, models, plots, and queues from scratch in under 2 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/AakashDubba/aakash-flyrank-ml-capstone.git
cd aakash-flyrank-ml-capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute the complete capstone pipeline
python scripts/build_capstone_notebook.py
```

The script automatically executes `work/notebooks/capstone.ipynb` top to bottom, performs the client-holdout split, trains all models, generates all plots in `docs/charts/` and `work/figures/`, and writes metrics to `work/outputs/model_results.json`.

---

## 📑 Completed Assignment Index

| Assignment | Description | Notebook Link |
|---|---|---|
| **ML-02** | Research Question & Provisional Lane | [`work/notebooks/w01_research_question.ipynb`](work/notebooks/w01_research_question.ipynb) |
| **ML-03** | Frame Your Lane as an ML Task | [`work/notebooks/w02_ml_task_framing.ipynb`](work/notebooks/w02_ml_task_framing.ipynb) |
| **ML-04** | Search Intelligence Data Contract | [`work/notebooks/w03_data_contract.ipynb`](work/notebooks/w03_data_contract.ipynb) |
| **ML-07** | Baseline Action Score & Top-10 Review | [`work/notebooks/w04_baseline_score.ipynb`](work/notebooks/w04_baseline_score.ipynb) |
| **ML-11** | Full Capstone ML Pipeline & Evaluation | [`work/notebooks/capstone.ipynb`](work/notebooks/capstone.ipynb) |
| **ML-12** | Demo Outline, Social Post, Employer Summary | Closing cells of [`work/notebooks/capstone.ipynb`](work/notebooks/capstone.ipynb) |

---

## 🏆 Acknowledgments & Data Credit

**Built on the FlyRank ML Internship dataset** ([https://flyrank.ai](https://flyrank.ai)).

We credit the FlyRank data science and engineering teams for releasing the pseudonymized search warehouse dataset and providing the pedagogical guidelines for honest, leak-free applied machine learning in enterprise search intelligence.
