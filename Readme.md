# DCDXRec

> Dual Conditional Diffusion for Generative Cross-Domain Recommendation via Disentangled Knowledge Transfer

DCDXRec is a dual-target cross-domain recommendation framework that uses conditional diffusion models to transfer user preferences across domains in a structured and non-linear manner. Unlike traditional linear augmentation methods, DCDXRec generates semantically coherent embeddings and improves recommendation quality in both domains simultaneously.

---

## Overview

The proposed framework addresses limitations of prior cross-domain recommendation approaches by combining:

- Bidirectional knowledge transfer between two domains
- Shared and domain-specific preference disentanglement
- Graph-based user/item representation learning
- Conditional diffusion for non-linear embedding generation
- Joint optimization for both source and target domains

---

## Supported Datasets

This project uses Amazon Review Data (2018) and Douban Dataset:
[Amazon Review Data (2018)](https://jmcauley.ucsd.edu/data/amazon_v2/index.html)
[Douban Dataset]([https://jmcauley.ucsd.edu/data/amazon_v2/index.html](https://www.google.com/url?sa=i&source=web&rct=j&url=https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information&ved=2ahUKEwiCsPy0g_-TAxWn8DgGHfuxBVYQy_kOegQIBBAD&opi=89978449&cd&psig=AOvVaw0NLKm4-cQuIhgs3yDkFggN&ust=1776863747143000))



Download both:

- ratings only
- metadata

For the following categories:

| Short Name | Dataset Name |
|-----------|--------------|
| Movie | Movies and TV |
| Music | CDs and Vinyl |
| Cell | Cell Phones and Accessories |
| Elec | Electronics |
| Book | Books |

---

## Directory Structure

```bash
DCDXRec/
├── datasets/
│   ├── raw/
│   │   └── Amazon/
│   │       ├── Movie/
│   │       ├── Music/
│   │       ├── Cell/
│   │       ├── Elec/
│   │       └── Book/
│   └── processed/
│
├── models/
├── utils/
├── filter.py
├── process.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository_url>
cd DCDXRec
pip install -r requirements.txt
```

Recommended environment:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- SciPy
- tqdm

---

## Dataset Preparation

Place raw downloaded files into:

```bash
datasets/raw/Amazon/
```

Example:

```bash
datasets/raw/Amazon/Movie/
datasets/raw/Amazon/Music/
datasets/raw/Amazon/Cell/
datasets/raw/Amazon/Elec/
datasets/raw/Amazon/Book/
```

---

## Data Preprocessing

Move into dataset directory:

```bash
cd datasets
```

### Step 1: Filtering Raw Data

Removes sparse users/items and converts ratings into implicit interactions.

```bash
python filter.py --domain Movie
python filter.py --domain Music
python filter.py --domain Cell
python filter.py --domain Elec
python filter.py --domain Book
```

Filtered outputs will be saved to:

```bash
datasets/processed/
```

### Step 2: Build Cross-Domain Pairs

Generate paired datasets for training.

```bash
python process.py --domains Movie Music
python process.py --domains Cell Elec
python process.py --domains Book Movie
python process.py --domains Book Music
```

---

## Training

Run the main training pipeline:

```bash
python main.py
```

You may modify hyperparameters inside the configuration or source files.

---

## Model Architecture

DCDXRec consists of four main components:

### 1. Graph Representation Learning

Learns user/item embeddings independently in each domain using LightGCN-style propagation.

### 2. Shared-Specific Decomposition

Each embedding is divided into:

- Shared transferable preferences
- Domain-specific preferences

### 3. Cross-Domain Conditional Diffusion

Learns bidirectional mappings such as:

- Movie → Music
- Music → Movie

through denoising diffusion models.

### 4. Dual Objective Optimization

Two complementary supervision paths are used:

- Real → Real
- Real → Augmented

This balances stable learning and generative diversity.

---

## Evaluation Metrics

The model is evaluated using:

- HR@10
- NDCG@10

---

## Experimental Results

### Amazon: Movie and Music

| Method | Movie HR | Movie NDCG | Music HR | Music NDCG |
|-------|----------|------------|----------|------------|
| CrossAug | 53.06 | 38.94 | 45.09 | 30.82 |
| DCDXRec | **54.02** | **39.83** | **46.12** | **31.34** |

### Amazon: Cell and Elec

| Method | Cell HR | Cell NDCG | Elec HR | Elec NDCG |
|-------|---------|-----------|---------|-----------|
| CrossAug | 34.46 | 22.71 | 34.75 | 22.86 |
| DCDXRec | **35.27** | **23.26** | **35.13** | **24.41** |

### Additional Results

| Dataset Pair | Method | HR | NDCG |
|-------------|--------|----|------|
| Book & Movie | CrossAug | 59.27 | 36.54 |
| Book & Movie | DCDXRec | **61.11** | **38.29** |
| Book & Music | CrossAug | 60.77 | 38.44 |
| Book & Music | DCDXRec | **61.37** | **39.81** |

---

## Hyperparameters

| Parameter | Value |
|----------|------|
| Embedding Size | 128 |
| Shared Dimension | 64 |
| Specific Dimension | 64 |
| GCN Layers | 2 |
| Diffusion Steps | 100 |
| Hidden Size | 256 |
| Batch Size | 1024 |
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 0.001 / 0.0005 |

---

## Data Processing Rules

- Ratings greater than or equal to 4 are treated as positive feedback
- Users/items with fewer than 5 interactions are removed
- Leave-one-out split:
  - Last interaction for testing
  - Second last for validation
  - Remaining interactions for training

---

## Quick Start

```bash
git clone <repository_url>
cd DCDXRec
pip install -r requirements.txt

cd datasets

python filter.py --domain Movie
python filter.py --domain Music

python process.py --domains Movie Music

cd ..
python main.py
```

---

## Acknowledgement

Benchmarks used in this work:

- Amazon Review Data (2018)
- Douban Dataset

Relevant prior works:

- CrossAug
- LightGCN
- BiTGCF
- DisenCDR
