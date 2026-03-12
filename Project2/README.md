# Project 2: Clustering & Unsupervised Learning and Intro to Multi-modal Models

## Overview

This project explores clustering techniques, unsupervised learning methods, and multi-modal models across three distinct domains:

1. **Part 1**: Steam Reviews - Product Analytics with Representations and Clustering
2. **Part 2**: Deep Learning and Clustering of Image Data (TensorFlow Flowers)
3. **Part 3**: Multi-modal Pokémon Classification with CLIP and VLMs

---

## Table of Contents

- [Requirements](#requirements)
- [Datasets](#datasets)
- [Part 1: Steam Reviews Analysis](#part-1-steam-reviews-analysis)
- [Part 2: Image Clustering with VGG Features](#part-2-image-clustering-with-vgg-features)
- [Part 3: Multi-modal Pokémon Classification](#part-3-multi-modal-pokémon-classification)
- [Key Findings](#key-findings)
- [File Structure](#file-structure)

---

## Requirements

### Python Libraries
```bash
pip install pandas numpy nltk scikit-learn transformers torch torchvision
pip install matplotlib seaborn tqdm umap-learn
pip install sentence-transformers
pip install open_clip_torch  # For CLIP
```

### Additional Dependencies
- NLTK data: `punkt`, `stopwords`
- Pre-trained models:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - VGG-16 (ImageNet pretrained)
  - `openai/clip-vit-base-patch32` or `ViT-B-32`
  - `Qwen/Qwen3-VL-2B-Instruct` (for VLM reranking)
  - `Qwen/Qwen3-4B-Instruct-2507` (for LLM labeling)

---

## Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| `main.csv` | Steam game reviews with game names, review text, and recommendations | 40,000 reviews |
| `heldout.csv` | Held-out game reviews for genre estimation | ~100 reviews |
| `Pokemon.csv` | Pokémon metadata with types | 753 Pokémon |
| `images/` | Pokémon images organized by name | 753 images |
| TensorFlow Flowers | Flower images (5 classes) | ~3,600 images |

---

## Part 1: Steam Reviews Analysis

### Task 1: Unsupervised Review Length Discovery

**Objective**: Create pseudo-labels based on review length and evaluate clustering performance.

#### Question 1: Define Pseudo-labels
- Tokenize each review using NLTK's `word_tokenize`
- Calculate 25th (Q25) and 75th (Q75) percentile token counts
- Label reviews as:
  - **Short**: token_count ≤ Q25
  - **Long**: token_count ≥ Q75
- Discard middle 50% of reviews

**Results**:
- Original: 40,000 reviews → Filtered: 20,676 reviews (51.69%)
- Short: 10,665 reviews (avg 7.37 tokens)
- Long: 10,011 reviews (avg 566.58 tokens)
- Length ratio: ~76.9× difference

#### Question 2: Feature Representations
Two representation methods implemented:

| Method | Dimensions | Type | Sparsity |
|--------|------------|------|----------|
| TF-IDF | ~25,000 | Sparse | ~99.97% |
| MiniLM | 384 | Dense | ~0.2% |

#### Questions 3-5: Clustering Pipelines

**Module Table**:

| Module | Alternatives | Hyperparameters |
|--------|--------------|-----------------|
| Representation | TF-IDF, MiniLM | min_df=3, stopwords='english' |
| Dimensionality Reduction | None, SVD, UMAP | n_components=50/200 |
| Clustering | K-Means, Agglomerative | k=2 |

**Best Pipeline**: **MiniLM + SVD(50) + Agglomerative**
- V-measure: 0.9773
- ARI: 0.9911
- Near-perfect separation of Short vs Long reviews

### Task 2: Game-Level Clustering

**Objective**: Cluster games based on aggregated positive review representations.

#### Question 6-7: Pipeline Construction
- Filter positive reviews (recommend=True)
- Aggregate reviews per game:
  - TF-IDF: Concatenate all reviews → compute TF-IDF
  - MiniLM: Average embeddings of all reviews

**Module Table** (24 total pipelines):

| Representation | Dim Reduction | Clustering |
|----------------|---------------|------------|
| TF-IDF, MiniLM | None, SVD(50), UMAP(50), Autoencoder(50) | K-Means(k=5), Agglomerative(n=5), HDBSCAN |

#### Question 8: Best Pipeline Selection
Using composite scoring (Purity 50%, Coverage 40%, Cluster Count 10%):

**Best Pipeline**: **MiniLM + SVD(50) + Agglomerative (n=5)**
- Average Cluster Purity: 84.53%
- Coverage: 100%
- 5 meaningful clusters with genre cohesion

### Task 3: Genre Estimation and Theme Clustering

#### Question 9-10: Genre Estimation
- Compute held-out game's embedding using positive reviews
- Assign to closest cluster from Task 2
- Report cluster's genre distribution as estimated genres

#### Question 11-12: Theme Clustering with LLM Labeling
- Cluster negative reviews (complaints) and positive reviews (praises) separately
- Use TF-IDF + SVD + K-Means for theme discovery
- Generate cluster labels using Qwen3-4B-Instruct:
  - Input: Top TF-IDF terms + exemplar reviews
  - Output: 3-6 word descriptive label

**Example Labels**:
- "Crashes on High-End PCs" (technical issues cluster)
- "Boring and Unfunny Combat Bosses" (gameplay complaints)

---

## Part 2: Image Clustering with VGG Features

### Overview
Cluster TensorFlow Flowers dataset (5 classes: daisy, dandelion, roses, sunflowers, tulips) using transfer learning.

### Feature Extraction (Q13-14)
- Use pre-trained VGG-16 network
- Extract features from fc6 layer (4096 dimensions)
- Dense features capture learned visual patterns

### Questions 15-16: Feature Comparison

| Feature Type | Dimensions | Sparsity | Memory |
|--------------|------------|----------|--------|
| Raw Pixels | 224×224×3 = 150,528 | Variable | High |
| VGG fc6 | 4,096 | Dense (~0%) | Moderate |
| TF-IDF (text) | ~25,000 | ~99.97% | Sparse storage |

### Question 17: t-SNE Visualization
- VGG features show clear cluster separation in 2D t-SNE projection
- Different flower types form distinct regions

### Question 18: Clustering Performance

**Module Table**:

| Dim Reduction | Clustering | Best ARI |
|---------------|------------|----------|
| None, SVD(50), UMAP(50), Autoencoder(50) | K-Means(k=5), Agglomerative(n=5), HDBSCAN | Varies |

### Question 19: MLP Classification
Compare classification performance on:
- Original VGG features (4096-dim)
- Reduced features (50-dim via SVD/UMAP/Autoencoder)

---

## Part 3: Multi-modal Pokémon Classification

### Setup
- **Dataset**: 753 Pokémon with images and type labels (18 types)
- **Model**: CLIP (ViT-B-32) for zero-shot classification

### Question 20: Text-to-Image Retrieval
Test different query templates for type retrieval:

```
Templates tested:
- "type: {type}"
- "{type} type Pokémon"
- "Pokémon with {type} abilities"
- "a {type} type creature"
```

**Results**:

| Type | Precision@5 |
|------|-------------|
| Bug | 100% |
| Fire | 100% |
| Grass | 80% |
| Dark | 40% |
| Dragon | 60% |

**Key Insight**: Types with clear visual correlations (Bug, Fire) perform best. Abstract types (Dark, Dragon) are harder for CLIP.

### Question 21: Image-to-Text Classification
- Encode Pokémon image
- Compare to embeddings of all type prompts
- Predict top-5 types

### Question 22: Overall Evaluation

| Metric | Value |
|--------|-------|
| **Acc@1** | 43.29% |
| **Hit@5** | 76.89% |
| Gap | 33.60% |

**Per-Type Performance**:
- Best: Dark (66.67%), Bug (63.77%), Ice (62.50%)
- Worst: Fighting (3.12%), Flying (0%), Ghost (12%)

### Question 23: VLM Reranking

**Approach**: Use Qwen3-VL-2B-Instruct to rerank CLIP's top-5 candidates.

**Results**:

| Metric | Value |
|--------|-------|
| CLIP Acc@1 | 43.29% |
| VLM-Reranked Acc@1 | 39.58% |
| **Change** | **-3.72%** |

**Conclusion**: VLM reranking **does not help** for this task. CLIP's direct similarity matching outperforms VLM-based reasoning.

---

## Key Findings

### Part 1: Text Representations
1. **MiniLM >> TF-IDF** for semantic clustering tasks (9× better V-measure)
2. **Dimensionality reduction matters**: SVD(50) + Agglomerative achieves near-perfect results
3. **UMAP can hurt**: Over-reduction destroys semantic signal

### Part 2: Image Features
1. **Transfer learning works**: VGG features generalize to flower classification
2. **Dense features** are fundamentally different from sparse TF-IDF
3. **Dimensionality reduction** can maintain clustering quality while reducing computation

### Part 3: Multi-modal Models
1. **CLIP excels at visually-grounded types** (Bug, Fire, Water)
2. **Abstract types are challenging** (Fighting, Ghost, Dark)
3. **Larger models ≠ better results**: Task-specific training matters more

---

## File Structure

```
Project 2/
├── Project2.ipynb          # Main notebook with all code and analysis
├── main.csv                # Steam reviews dataset
├── heldout.csv             # Held-out game for genre estimation
├── Pokemon.csv             # Pokémon metadata
├── images/                 # Pokémon images directory
│   ├── Abomasnow/
│   ├── Abra/
│   └── ...
├── helper-code.ipynb       # Utility functions
├── pokedex_helper.ipynb    # Pokémon data preprocessing
└── README.md               # This file
```

---

## Reproducibility Notes

1. **Random Seeds**: Set `random_state=42` for reproducibility
2. **GPU Required**: MiniLM, CLIP, and VLM inference benefit from GPU acceleration
3. **Google Colab**: Original code designed for Colab with Google Drive integration
4. **Model Downloads**: Pre-trained models will be downloaded automatically on first run

---

## Authors

ECE 219 - Machine Learning for Large-Scale Data Mining  
UCLA Winter 2026
