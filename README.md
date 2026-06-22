# Cluster Feature Selector — Random Forest + TreeSHAP

> **Given any tabular dataset with cluster labels, this tool identifies the top-3 features that best distinguish a target cluster from all others — using a Random Forest classifier explained by TreeSHAP.**

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Toponymy Integration](#toponymy-integration)
3. [MITRE ATT&CK Integration](#mitre-attck-integration)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Prepare Your Dataset](#prepare-your-dataset)
7. [Quick Start](#quick-start)
8. [Usage as a Python Function](#usage-as-a-python-function)
9. [Usage from the Command Line](#usage-from-the-command-line)
10. [Function Reference](#function-reference)
11. [Understanding the Output](#understanding-the-output)
12. [How It Works](#how-it-works)
13. [Troubleshooting](#troubleshooting)

---

## What It Does

`top-cluster-features-selector-using-treeshap.py` trains a **one-vs-all binary Random Forest** to separate a chosen cluster from all other clusters, then runs **TreeSHAP** to rank every feature by how much it influences that prediction. The result is the **top-K feature names** (default: 3) that most strongly characterise the target cluster.

```
Your CSV  →  Binary RF (cluster N vs. rest)  →  TreeSHAP  →  ["feat_a", "feat_b", "feat_c"]
```

---

## Toponymy Integration

This tool is designed to plug directly into a **Toponymy pipeline** — the process of automatically assigning meaningful, human-readable names to machine-generated clusters.

Once the top-3 discriminating features are extracted, they can be passed as a prompt to any LLM to generate a descriptive cluster name:

```python
from cluster_shap_rf import get_top_features
import anthropic

# Step 1 — extract the top features for the target cluster
top3 = get_top_features("your_data.csv", cluster_id=0, verbose=False)

# Step 2 — ask an LLM to name the cluster based on those features
client  = anthropic.Anthropic()
prompt  = (
    f"I have a cluster of network events characterised by these three "
    f"discriminating features: {top3}. "
    f"Suggest a concise, descriptive name for this cluster that captures "
    f"its behaviour or threat profile."
)
message = client.messages.create(
    model      = "claude-opus-4-6",
    max_tokens = 256,
    messages   = [{"role": "user", "content": prompt}]
)
cluster_name = message.content[0].text
print(f"Suggested cluster name: {cluster_name}")
```

The full four-stage Toponymy pipeline looks like this:

```
Stage 1 — Represent   →  vectorise raw telemetry (TF-IDF, embeddings, …)
Stage 2 — Cluster     →  group samples (K-Means, HDBSCAN, …)
Stage 3 — Characterise→  [THIS TOOL] extract top-3 TreeSHAP features per cluster
Stage 4 — Name        →  pass features to an LLM → human-readable cluster label
```

> **Why only 3 features?**
> LLMs produce sharper, more grounded names when given a small set of highly discriminating signals rather than a long undifferentiated list. Three features strike the right balance between specificity and conciseness.

---

## MITRE ATT&CK Integration

> ⚠️ **Important caveat — when raw features are not enough**

In cybersecurity datasets (such as DARPA OpTC process-tree telemetry), the top-3 features returned by TreeSHAP are often **low-level technical fields** — e.g. `parent_pid`, `image_load_count`, `registry_write_count`. Feeding these directly to an LLM may produce cluster names that are technically accurate but **not meaningful** to threat analysts.

**Recommended solution: ground the features in MITRE ATT&CK before naming.**

The idea is to map each discriminating feature value to one or more ATT&CK techniques or tactics, then pass *those* mappings to the LLM instead of (or alongside) the raw feature names:

```
top-3 features
      │
      ▼
MITRE ATT&CK lookup
(feature value → Technique ID + Tactic)
      │
      ▼
LLM prompt enriched with ATT&CK context
      │
      ▼
Cluster name grounded in threat intelligence
(e.g. "Credential Access via LSASS Memory Dumping")
```

### Example enriched prompt

```python
from cluster_shap_rf import get_top_features

top3 = get_top_features("your_data.csv", cluster_id=0, verbose=False)

# Manually or programmatically map features to ATT&CK context
# (use the MITRE ATT&CK STIX API or attackcti library for automation)
attack_context = {
    "registry_write_count": "T1112 — Modify Registry (Defense Evasion)",
    "image_load_count":     "T1129 — Shared Modules (Execution)",
    "parent_process_name":  "T1059 — Command and Scripting Interpreter (Execution)",
}

enriched = [
    f"{feat}  →  {attack_context.get(feat, 'unknown technique')}"
    for feat in top3
]

prompt = (
    "I have a cluster of process-tree events. "
    "The most discriminating features and their MITRE ATT&CK mappings are:\n"
    + "\n".join(f"  • {e}" for e in enriched)
    + "\nSuggest a concise threat-actor behaviour name for this cluster."
)
```

### Automating ATT&CK lookups

The [`attackcti`](https://github.com/OTRF/ATTACK-Python-Client) library provides a Python client for the official MITRE ATT&CK STIX feed:

```bash
pip install attackcti
```

```python
from attackcti import attack_client

client     = attack_client()
techniques = client.get_techniques()
# filter by keyword matching your feature names / values
```

### When to use ATT&CK enrichment

| Situation | Recommendation |
|---|---|
| Features are **semantic** (e.g. `attack_type`, `protocol`, `service`) | Raw features → LLM is usually sufficient |
| Features are **low-level numeric** (e.g. `pid`, `byte_count`, `load_count`) | Always enrich with ATT&CK before naming |
| Dataset is **network flow** or **process-tree telemetry** | Strongly recommended to use ATT&CK enrichment |
| Dataset is **non-security** (e.g. finance, healthcare) | ATT&CK enrichment not applicable; use domain ontology instead |

---

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| scikit-learn | ≥ 1.2 |
| shap | ≥ 0.41 |
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |

---

## Installation

### 1 — Clone or download the file

```bash
# Option A — clone the whole repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Option B — download just the script
curl -O https://raw.githubusercontent.com/your-username/your-repo/main/cluster_shap_rf.py
```

### 2 — Install Python dependencies

```bash
pip install scikit-learn shap pandas numpy
```

> **Using a virtual environment (recommended)**
> ```bash
> python -m venv .venv
> source .venv/bin/activate        # Windows: .venv\Scripts\activate
> pip install scikit-learn shap pandas numpy
> ```

---

## Prepare Your Dataset

Your CSV (or DataFrame) must follow two rules:

| Rule | Detail |
|---|---|
| **Cluster column** | One column must hold integer cluster IDs. Supported column names (detected automatically): `cluster`, `Cluster`, `cluster_id`, `ClusterID`, `cluster_label`, `label`, `Label`, `y`. |
| **Feature columns** | All other columns are treated as features. They can be numeric or categorical — the tool encodes categoricals automatically. |

### Minimal example

```
feature_1, feature_2, feature_3, cluster
0.12,      5,         "tcp",     0
0.98,      2,         "udp",     1
0.45,      8,         "tcp",     0
...
```

> **Missing values?** Filled with `0` automatically.  
> **Constant columns?** Dropped automatically.

---

## Quick Start

Place your CSV in the **same folder** as `cluster_shap_rf.py`, then run:

```bash
python cluster_shap_rf.py --dataset your_data.csv --cluster 0
```

You will see a progress log and a final result like:

```
=================================================================
  TOP 3 FEATURES — Cluster 0
=================================================================
  #1  duration                                    
  #2  src_bytes                                   
  #3  dst_bytes                                   
=================================================================

Result: ['duration', 'src_bytes', 'dst_bytes']
```

---

## Usage as a Python Function

Import `get_top_features` into any Python script or notebook.

### Basic call

```python
from cluster_shap_rf import get_top_features

top3 = get_top_features("your_data.csv", cluster_id=0)
print(top3)
# ['duration', 'src_bytes', 'dst_bytes']
```

### Pass an already-loaded DataFrame

```python
import pandas as pd
from cluster_shap_rf import get_top_features

df   = pd.read_csv("your_data.csv")
top3 = get_top_features(df, cluster_id=0)
print(top3)
```

### Target a different cluster

```python
# Explain what makes cluster 3 unique
top3 = get_top_features("your_data.csv", cluster_id=3)
```

### Retrieve more (or fewer) top features

```python
# Return the top 5 features instead of 3
top5 = get_top_features("your_data.csv", cluster_id=0, top_k=5)
```

### Silent mode (no console output)

Useful when embedding inside a larger pipeline.

```python
top3 = get_top_features("your_data.csv", cluster_id=0, verbose=False)
```

### Loop over all clusters

```python
import pandas as pd
from cluster_shap_rf import get_top_features

df       = pd.read_csv("your_data.csv")
clusters = sorted(df["cluster"].unique())

results = {}
for cid in clusters:
    results[cid] = get_top_features(df, cluster_id=cid, verbose=False)
    print(f"Cluster {cid}: {results[cid]}")
```

### Use inside a Jupyter / Marimo notebook

```python
from cluster_shap_rf import get_top_features

top3 = get_top_features("your_data.csv", cluster_id=1, verbose=True)
```

---

## Usage from the Command Line

```bash
# Default — cluster 0, top 3 features
python cluster_shap_rf.py

# Specify dataset path
python cluster_shap_rf.py --dataset /path/to/your_data.csv

# Target a different cluster
python cluster_shap_rf.py --dataset your_data.csv --cluster 2

# Return top 5 features
python cluster_shap_rf.py --dataset your_data.csv --cluster 0 --top-k 5

# Suppress all output except the final result
python cluster_shap_rf.py --dataset your_data.csv --cluster 0 --quiet
```

### All CLI flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset` | string | `DARPA_OpTC_50k_ProcessTree-2.csv` | Path to the CSV file |
| `--cluster` | int | `0` | Target cluster ID |
| `--top-k` | int | `3` | Number of top features to return |
| `--quiet` | flag | off | Suppress verbose output |

---

## Function Reference

```python
get_top_features(
    dataset    : str | Path | pd.DataFrame,
    cluster_id : int  = 0,
    top_k      : int  = 3,
    verbose    : bool = True,
) -> list[str]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | `str`, `Path`, or `pd.DataFrame` | — | CSV file path **or** a DataFrame already in memory |
| `cluster_id` | `int` | `0` | The cluster whose features you want to explain |
| `top_k` | `int` | `3` | How many top features to return |
| `verbose` | `bool` | `True` | Print step-by-step progress and evaluation metrics |

**Returns** `list[str]` — feature names ordered from most to least discriminating.

**Raises**
- `FileNotFoundError` — if the given file path does not exist.
- `ValueError` — if `cluster_id` is not present in the cluster column.

---

## Understanding the Output

### Console log (verbose=True)

```
=================================================================
  get_top_features | cluster_id=0 | top_k=3
=================================================================
  Dataset shape  : (50000, 24)
  Cluster column : 'cluster'

  Cluster distribution:
  0    18200
  1     9400
  2    22400

  Target         : Cluster 0
  In-cluster     : 18200 samples  (label = 1)
  Out-of-cluster : 31800 samples  (label = 0)
  Features : 23  |  Train : 40000  |  Test : 10000

  Training Random Forest (Cluster 0 vs. Rest) ...
  Done — 300 trees built.

  Evaluation:
  ROC-AUC : 0.9741

                  precision  recall  f1-score  support
  Other Clusters   0.9600    0.9712    0.9656     6360
  Cluster 0        0.9401    0.9198    0.9298     3640

  TreeSHAP feature ranking:
  Rank   Feature                                    Mean |SHAP|
  ──────────────────────────────────────────────────────────────
  1      duration                                      0.043210  ◄
  2      src_bytes                                     0.031874  ◄
  3      dst_bytes                                     0.028901  ◄
  4      protocol_type                                 0.017233
  ...

=================================================================
  TOP 3 FEATURES — Cluster 0
=================================================================
  #1  duration
  #2  src_bytes
  #3  dst_bytes
=================================================================
```

### What the numbers mean

| Term | Meaning |
|---|---|
| **ROC-AUC** | How well the model separates the target cluster from the rest. Closer to 1.0 is better. |
| **Mean \|SHAP\|** | Average absolute Shapley value across all test samples. Higher = more influential for distinguishing this cluster. |

---

## How It Works

```
Step 1 — Load & Preprocess
        Detect cluster column → build binary label (1 = target, 0 = rest)
        Label-encode categoricals → fill NaN → 80/20 stratified split

Step 2 — Train Random Forest
        300 trees, balanced class weights (handles imbalanced clusters),
        trained on 80 % of the data

Step 3 — Evaluate
        ROC-AUC + classification report on the held-out 20 % test set

Step 4 — TreeSHAP
        shap.TreeExplainer computes exact Shapley values for every test
        sample → features ranked by mean absolute SHAP value

Step 5 — Return top-K
        The K feature names with the highest mean |SHAP| are returned
```

> **Why TreeSHAP?**  
> Unlike simple feature importance (which is biased toward high-cardinality features), SHAP values are grounded in game theory and measure each feature's *actual contribution* to individual predictions, averaged across the entire test set.

---

## Troubleshooting

### `FileNotFoundError: Dataset not found`
Make sure the CSV path is correct. Use an absolute path if in doubt:
```python
get_top_features("/absolute/path/to/your_data.csv", cluster_id=0)
```

### `ValueError: Cluster 5 not found`
The cluster ID you passed does not exist. Check available IDs:
```python
import pandas as pd
df = pd.read_csv("your_data.csv")
print(df["cluster"].unique())
```

### `No cluster column found — using last column`
Your cluster column has a non-standard name. Rename it to one of the supported names (`cluster`, `label`, etc.) before calling the function:
```python
df = df.rename(columns={"my_custom_cluster_col": "cluster"})
get_top_features(df, cluster_id=0)
```

### SHAP takes too long on a large dataset
Pass a pre-sampled test subset to speed things up:
```python
import pandas as pd
from cluster_shap_rf import get_top_features

df      = pd.read_csv("your_data.csv")
sample  = df.sample(n=5000, random_state=42)   # work on 5 000 rows
top3    = get_top_features(sample, cluster_id=0)
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
