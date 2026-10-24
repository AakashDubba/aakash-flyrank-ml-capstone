# Capstone Report — Prioritizing Content Refresh Opportunities in Enterprise SEO

- **Author:** Aakash Dubba
- **Lane:** Refresh / Content Opportunity Scoring (Lane 2)
- **Repo:** [https://github.com/AakashDubba/aakash-flyrank-ml-capstone](https://github.com/AakashDubba/aakash-flyrank-ml-capstone)
- **Deployed Paper URL:** [https://aakashdubba.github.io/aakash-flyrank-ml-capstone/](https://aakashdubba.github.io/aakash-flyrank-ml-capstone/)
- **Date:** October 2026

---

## 0. Abstract

1. **Question:** In enterprise organic search operations, editorial and SEO teams face thousands of decaying content URLs but lack principled, data-driven methods to identify which pages warrant high-effort content refresh versus monitoring or consolidation.
2. **Data:** Using the FlyRank ML Internship dataset comprising 30,000 pseudonymized content assets across 32 enterprise clients (with a 54.2% base decline rate), we evaluated observable search performance, query engagement, and content age signals without using titles, domains, or search queries.
3. **Method:** We established a strict group-aware validation design (client-holdout split holding out 20% of client portfolios, evaluating on 2,325 blind holdout assets) and benchmarked regularized linear, decision tree, and ensemble tree models against an interpretable multi-signal heuristic baseline while strictly enforcing zero target leakage.
4. **Headline Result:** The Random Forest classifier demonstrated decisive ranking superiority on unseen enterprise clients, achieving a Precision@20 of 0.800 (a 3.20× lift over the baseline's 0.250) and a Precision@50 of 0.760 (a 4.22× lift over the baseline's 0.180), with an out-of-fold ROC-AUC of 0.751 and Average Precision (PR-AUC) of 0.622.
5. **Decision Support:** Blending model-predicted decline probability with verified traffic impact yields a prioritized editorial review queue with explainable reason codes, reducing manual auditing fatigue while focusing scarce writer capacity on reversible, high-yield organic assets.

---

## 1. Problem Framing

### Operational Decision & Stakeholders
Content refresh in enterprise environments is resource-constrained. Rewriting an outdated technical guide, conducting updated expert interviews, and optimizing structural SERP schema requires between 4 to 12 hours of editorial labor per URL. When marketing portfolios exceed 10,000 to 50,000 indexed pages, editorial sprint planning collapses under volume. Editors often resort to simplistic rules—such as reviewing any page older than six months—which misallocates human attention to dead topics or healthy ranking assets.

This research directly answers:
> *"Which declining, high-visibility content pages should an editorial and SEO team prioritize for content refresh, expansion, or consolidation to maximize search traffic recovery while minimizing wasted editorial effort?"*

### Unit of Analysis & Grain
- **Unit of Analysis:** Individual content asset (`content_id` grain), aggregated over a stable 90-day historical window.
- **Decision Makers:** SEO Directors, Content Marketing Leads, and Managing Editors.
- **Operational Output:** A calibrated **Opportunity Score (0–100)**, an out-of-sample predicted probability of traffic decline, and discrete action classifications with auditable reason codes:
  - `refresh_and_review_ctr`: Deep editorial refresh paired with meta title and SERP snippet optimization for high-impression pages with sub-optimal click-through rates.
  - `refresh_and_review_engagement`: On-page UX improvement, structural reorganization, and readability expansion for pages experiencing high bounce or low scroll depth.
  - `expand_and_refresh`: Comprehensive topical expansion for high-intent, high-competition keywords.
  - `monitor`: Low-touch automated monitoring for low-demand or stable assets.

### Cost of a Wrong Call
- **False Positive (Type I Error):** Recommending a dead or irrecoverable page for refresh wastes an entire writer-day (8+ hours), delivering negligible organic traffic upside.
- **False Negative (Type II Error):** Overlooking a high-traffic revenue driver in early decay allows competitors to permanently seize top SERP rankings, resulting in severe long-term customer acquisition loss.

---

## 2. Data Safety & Data Contract

### Source & Grains
We conducted our experiments on the bundled slice of the FlyRank pseudonymized warehouse release (`data/raw/content_refresh_anonymized.csv`), comprising:
- **Rows:** 30,000 distinct content pages
- **Clients:** 32 enterprise client organizations
- **Raw Features:** 44 observable columns
- **Temporal Window:** 90-day aggregation window, with 30-day split sub-windows (`impressions_last_30d` vs `impressions_prev_30d`) representing trailing momentum.

### Privacy Safeguards & Data Exclusions
In strict adherence to `DATA_USE.md` and enterprise privacy standards:
1. **Identifiable Columns Excluded:** No raw URLs, URL slugs, page titles, client domain names, client brand names, or private search queries were ingested or analyzed.
2. **Group Identifiers Restricted:** `client_id` was retained exclusively for group-aware splitting and cluster analysis; it was strictly prohibited from being passed to models as a predictive feature.
3. **Leakage Elimination:** Columns directly encoding the outcome—specifically `trend_direction` and `trend_pct`—were purged prior to feature compilation.

### Missingness & Imputation Contract
- Numeric missing values (such as `cpc` or missing engagement fields) were imputed using robust domain medians or zeros for event counts.
- Categorical attributes with missing labels were mapped to an explicit `"unknown"` category to preserve missingness informativeness without data loss.

---

## 3. Baseline Model

### Transparent Heuristic Rule
Before training any machine learning models, we formulated an interpretable multi-signal business heuristic mirroring the best practices of senior SEO strategists:
$$\text{Baseline Score} = \left( 0.35 \times \text{Norm}(\text{impressions}_{90d}) + 0.25 \times \text{Norm}(\text{staleness}) + 0.20 \times \mathbb{I}(\text{CTR} < 1.0\%) + 0.20 \times \mathbb{I}(\text{pos} \in [4, 20]) \right) \times 100$$

### Baseline Performance
Evaluated on the exact same client-holdout test set (2,325 assets from 6 unseen clients), the baseline rule achieved:
- **ROC-AUC:** 0.592
- **PR-AUC (Average Precision):** 0.435 (below the 0.542 base rate)
- **Precision@10:** 0.300
- **Precision@20:** 0.250
- **Precision@50:** 0.180
- **Precision@100:** 0.210
- **Brier Score:** 0.240

**Baseline Diagnosis:** While heuristic rules successfully isolate pages with high impression volume and aging publication dates, they fail to disentangle whether high-volume older pages are actually decaying or functioning as stable evergreen pillars. Consequently, the top-20 recommendations from hand-crafted rules include an unacceptably high rate of false positives (75% non-declining content).

---

## 4. Model Architecture & Feature Engineering

### Feature Transformation
From the raw observable data, we compiled 18 numerical and 8 categorical features:
- **Logarithmic Transforms:** High-skew count signals (`impressions_90d`, `clicks_90d`, `sessions_90d`, `ai_sessions_90d`) were log-transformed ($\log(1+x)$) to stabilize variance.
- **Behavioral Ratios:** `ctr`, `engagement_rate`, `scroll_rate`, and `ai_traffic_pct`.
- **Temporal Durations:** `content_age_days` and `days_since_last_update`.
- **One-Hot Encoding:** Categorical tiers (`competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`).

### Target Definition
$$\text{is\_declining\_label} = \mathbb{I}(\text{trend\_direction} == \text{"down"})$$
- Target positive rate (base rate): **54.21%** (16,262 declining assets out of 30,000).

### Evaluated Model Families
1. **L2 Regularized Logistic Regression:** Standardized features, balanced class weights, serving as the linear benchmark.
2. **Decision Tree Classifier:** Depth-constrained (`max_depth=5`, `min_samples_leaf=50`) non-linear tree.
3. **Random Forest Classifier:** Ensembled bagging architecture (`n_estimators=200`, `max_depth=10`, `min_samples_leaf=25`, `class_weight='balanced_subsample'`).
4. **Gradient Boosting Classifier:** Sequential boosting minimizing log-loss (`n_estimators=100`, `max_depth=4`, `learning_rate=0.08`).

---

## 5. Evaluation & Validation Design

### Group-Aware Client-Holdout Split
A central failure mode in SEO machine learning is **domain leakage**. Because pages within a single domain share identical backlink profiles, CMS templates, author authority, and brand awareness, traditional random row splits leak domain-level signals between training and test sets.

To enforce production-grade validity:
- We grouped all 30,000 pages by `client_id`.
- Exactly 20% of client portfolios (6 unique client domains comprising 2,325 pages) were held out as a blind test set.
- Exactly 80% of client portfolios (26 client domains comprising 27,675 pages) were used for model fitting.
- **Zero Client Overlap:** Verified $\text{Clients}_{\text{train}} \cap \text{Clients}_{\text{test}} = \emptyset$.

### Comparative Results Table (Identical Client-Holdout Split)

| Model Architecture | Precision@10 | Precision@20 | Precision@50 | Precision@100 | ROC-AUC | PR-AUC | Brier Score | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Random Chance (Base Rate)** | 0.542 | 0.542 | 0.542 | 0.542 | 0.500 | 0.542 | 0.250 | 1.000 | 0.703 |
| **Heuristic Baseline** | 0.300 | 0.250 | 0.180 | 0.210 | 0.592 | 0.435 | 0.240 | 0.020 | 0.036 |
| **Logistic Regression (L2)** | 0.200 | 0.350 | 0.420 | 0.450 | 0.700 | 0.523 | 0.216 | 0.567 | 0.566 |
| **Decision Tree (Depth 5)** | 0.800 | 0.800 | 0.680 | 0.650 | 0.742 | 0.575 | 0.200 | 0.716 | 0.634 |
| **Random Forest (Best General)** | **0.700** | **0.800** | **0.760** | **0.730** | **0.751** | **0.622** | **0.198** | **0.741** | **0.643** |
| **Gradient Boosting** | **0.800** | **0.850** | **0.820** | **0.850** | **0.778** | **0.678** | **0.192** | **0.755** | **0.648** |

### Key Findings on Precision@K
1. **Dramatic Lift Over Baseline:** The Random Forest model achieved a **Precision@20 of 0.800**, representing a **3.20× lift** over the heuristic baseline (0.250). At $K=50$, Random Forest achieved **0.760 vs. 0.180 (a 4.22× lift)**.
2. **Base Rate Superiority:** While the heuristic baseline degraded well below the 54.2% task base rate in its top ranks, the tree ensemble maintained strong discrimination (73% to 80% true positive rate in top cohorts).
3. **Calibration:** Random Forest achieved a Brier score of 0.198, demonstrating reliable probability estimates for downstream decision thresholds.

---

## 6. Interpretation & Feature Drivers

### Top Feature Importances (Mean Decrease in Impurity)
The top 10 predictive features in the Random Forest model:
1. `days_with_impressions` (14.66%): Impression continuity across the 90-day window is the single strongest indicator of organic stability vs. decay.
2. `log_impressions_90d` (12.45%): Magnitude of overall search exposure.
3. `avg_position` (11.44%): Average ranking position across search queries.
4. `content_age_days` (8.70%): Total lifespan of the article since initial indexing.
5. `char_count` (4.46%): Overall substantive depth of the written content.
6. `age_tier_365+` (4.19%): Categorical indicator for content older than 1 year.
7. `log_clicks_90d` (3.77%): Historical click capture volume.
8. `ctr` (3.64%): Observed SERP click-through rate relative to position expectations.
9. `word_count` (3.59%): Article body length.
10. `scroll_rate` (3.50%): User reader completion percentage.

### Surprises & Negative Findings
- **Staleness Alone is Not Predictive:** Simple `days_since_last_update` contributed only ~2.1% importance when isolated. High-ranking evergreen reference pages often go 400+ days without updates while preserving rock-solid organic visibility. Flagging pages based solely on age generates overwhelming false positive volume.
- **Continuity Beats Volume:** A page generating 100 impressions every day for 90 days (`days_with_impressions = 90`) is dramatically less likely to be in decline than a page with 15,000 impressions concentrated across only 10 days of fleeting trend spikes.

---

## 7. Action Recommendations & Editorial Playbook

### Calibrated Composite Opportunity Score
To support operational triage, we synthesize the model's calibrated probability of decline with business exposure into a single 0–100 **Opportunity Score**:
$$\text{Opportunity Score} = 0.50 \times \text{P}(\text{Decline}) + 0.30 \times \text{PercentileRank}(\log(\text{impressions}_{90d})) + 0.20 \times \text{PercentileRank}(\text{staleness})$$

### Top 10 Portfolio Opportunities Preview

| Rank | Content ID | Opp. Score | RF Prob. | Recommended Action | Diagnostic Reason Codes | Impressions | Position |
|---:|---|---:|---:|---|---|---:|---:|
| 1 | `content_ac1d924c6a70` | 85.98 | 0.755 | `refresh_and_review_engagement` | `model_decline_risk\|declining_with_demand\|low_engagement_visible_page` | 21,853 | 31.7 |
| 2 | `content_1bfaa38ff26c` | 85.16 | 0.732 | `refresh_and_review_engagement` | `model_decline_risk\|declining_with_demand\|low_engagement_visible_page` | 25,715 | 22.2 |
| 3 | `content_bffd32d0b4b1` | 85.11 | 0.772 | `expand_and_refresh` | `model_decline_risk\|declining_with_demand` | 10,910 | 37.7 |
| 4 | `content_18d83d6ef368` | 83.74 | 0.758 | `refresh_and_review_ctr` | `model_decline_risk\|declining_with_demand\|low_ctr_visible_page` | 8,864 | 20.0 |
| 5 | `content_0942ed0b1ec1` | 83.33 | 0.738 | `refresh_and_review_engagement` | `model_decline_risk\|declining_with_demand\|low_engagement_visible_page` | 10,744 | 24.4 |
| 6 | `content_541cec39d61b` | 83.30 | 0.765 | `expand_and_refresh` | `model_decline_risk\|declining_with_demand` | 7,060 | 41.1 |
| 7 | `content_7368877ea310` | 83.24 | 0.675 | `refresh_and_review_engagement` | `model_decline_risk\|declining_with_demand\|low_engagement_visible_page` | 59,472 | 24.8 |
| 8 | `content_cac9eaac0184` | 83.16 | 0.763 | `expand_and_refresh` | `model_decline_risk\|declining_with_demand` | 6,974 | 30.8 |
| 9 | `content_cf56e2e2e282` | 82.98 | 0.669 | `refresh_and_review_ctr` | `model_decline_risk\|declining_with_demand\|low_ctr_visible_page\|low_engagement_visible_page` | 61,678 | 19.7 |
| 10 | `content_5feee3994adb` | 82.67 | 0.743 | `expand_and_refresh` | `model_decline_risk\|declining_with_demand` | 7,812 | 39.0 |

### Editorial Workflow Integration
- **Weekly Sprint Limit:** Content leads should ingest only the top 20–50 items per client portfolio per sprint, matching real writer bandwidth.
- **Triage Matrix:**
  - `refresh_and_review_ctr`: Assign to SEO copywriters to rewrite Title tags and Meta descriptions for snippet re-indexing.
  - `refresh_and_review_engagement`: Assign to UX and subject matter experts to add diagrams, TL;DR executive summaries, and interactive components.
  - `expand_and_refresh`: Assign to senior writers for comprehensive 1,500+ word topical depth additions.

---

## 8. Reproducibility & Environment

### Step-by-Step Re-run Instructions
To reproduce all numbers, models, and artifacts from a fresh clone:
```bash
git clone https://github.com/AakashDubba/aakash-flyrank-ml-capstone.git
cd aakash-flyrank-ml-capstone
pip install -r requirements.txt
python scripts/build_capstone_notebook.py
```
This single execution script automatically rebuilds `work/notebooks/capstone.ipynb`, performs the strict client-holdout split, evaluates all model pipelines, exports high-resolution SVG/PNG charts, and generates the final receipts in `work/outputs/model_results.json`.

### Verification Artifacts
- **Random Seed:** Set globally to `42` (`np.random.seed(42)` and `random_state=42` across all pipelines).
- **Metric Receipts:** Committed under `work/outputs/model_results.json`.
- **Ranked Output:** Committed under `work/outputs/final_refresh_queue.csv`.

---

## 9. Acknowledgments & Data Credit

**Built on the FlyRank ML Internship dataset** ([https://flyrank.ai](https://flyrank.ai)).

We express our gratitude to the FlyRank machine learning and data engineering teams for open-sourcing the curated pseudonymized search warehouse release, maintaining robust baseline standards, and providing pedagogical frameworks for honest applied machine learning in information retrieval.
