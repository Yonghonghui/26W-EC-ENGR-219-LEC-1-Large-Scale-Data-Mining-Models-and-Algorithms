# ECE 219 Project 1: Text Mining and Semantic Analysis

![UCLA](https://img.shields.io/badge/UCLA-ECE%20219-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/Transformers-4.30%2B-yellow)

## 📖 Project Overview

This project is developed for **ECE 219: Large Scale Data Mining** (Winter 2026) at UCLA. The primary objective is to build a comprehensive text mining pipeline that evolves from traditional statistical methods to state-of-the-art Deep Learning and Large Language Model (LLM) approaches.

The project is divided into two major phases:
1.  **Text Classification & Clustering:** Analyzing high-dimensional text data from the *20 Newsgroups* dataset using dimensionality reduction and classic machine learning classifiers.
2.  **Semantic Paraphrase Detection:** Solving the complex task of identifying semantic equivalence using the *PAWS* (Paraphrase Adversaries from Word Scrambling) dataset, utilizing Transformer-based architectures and Prompt Engineering.

---

## 📂 Project Structure

The codebase is consolidated in `Project1.ipynb` and follows the structure below:

### Part 1: Text Processing & Representation
* **Data Ingestion:** Loading and parsing the *20 Newsgroups* dataset.
* **Preprocessing:** Tokenization, stop-word removal, and lemmatization.
* **Feature Engineering:** Implementing **TF-IDF** (Term Frequency-Inverse Document Frequency) to vectorize text documents.

### Part 2: Dimensionality Reduction
* **LSI (Latent Semantic Indexing):** Applying Singular Value Decomposition (SVD) to reduce feature space.
* **NMF (Non-negative Matrix Factorization):** Extracting latent topics from the document-term matrix.
* **Visualization:** Using 2D plots to visualize class separation after reduction.

### Part 3: Traditional Classification Models
Comparative analysis of multi-class classifiers with Hyperparameter Tuning (Grid Search):
* **Support Vector Machines (SVM):** Linear vs. Soft Margin.
* **Logistic Regression:** L1 (Lasso) vs. L2 (Ridge) Regularization.
* **Naive Bayes:** Multinomial implementation.

### Part 4: Deep Learning & LLMs (Advanced)
Focusing on the *PAWS* dataset to detect paraphrases:
* **Q16 (GloVe Baseline):** Static word embeddings averaged for sentence representation.
* **Q17 (Transformers):**
    * **(a) Bi-Encoder:** Independent sentence encoding using `sentence-transformers` (SBERT).
    * **(b) Cross-Encoder:** Fine-tuning `DistilBERT` for joint sequence-pair classification (SOTA performance).
* **Q18 (LLM Prompting):**
    * **Zero-shot & Few-shot Inference:** Evaluating instruction-tuned LLMs (e.g., `Qwen2.5-3B-Instruct` or `Llama-3`) without gradient updates.

---

## 📊 Key Results (Paraphrase Detection)

A comparison of different paradigms for semantic similarity (Q17 & Q18):

| Model Architecture | Approach | Test Accuracy | F1 Score | Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **GLoVe Baseline** | Static Embeddings | ~56.0% | ~32.1% | Fast, but lacks context |
| **Bi-Encoder** | Feature-based | 61.6% | 53.4% | Efficient for retrieval |
| **LLM (Zero-shot)** | Prompting | 77.5% | 74.6% | No training required |
| **Cross-Encoder** | Fine-tuning | **86.0%** | **84.5%** | **Highest Accuracy** |

> **Insight:** The **Cross-Encoder** significantly outperforms other methods by modeling the deep interaction between sentence pairs using full self-attention. The **LLM** demonstrates impressive generalization capabilities via simple prompting, while the **Bi-Encoder** offers a trade-off between speed and accuracy.

---

## 🛠️ Installation & Requirements

This project is optimized for **Google Colab** (T4 GPU recommended for Part 4).

### Dependencies
To install the necessary libraries, run the following command:

```bash
# General Data Science
pip install numpy pandas scikit-learn matplotlib seaborn nltk

# Deep Learning & NLP
pip install torch torchvision torchaudio
pip install transformers sentence-transformers datasets accelerate bitsandbytes
pip install umap-learn


## 🚀 How to Run

1.  **Download:** Clone this repository or download the `Project1.ipynb` file.
2.  **Environment:** Open the notebook in Google Colab or a local Jupyter environment with GPU support.
3.  **Execution:** Run the cells sequentially.
    * **Note:** For Q17 and Q18, ensure the runtime is set to **GPU** to speed up Transformer training and LLM inference.
    * **Note:** The *20 Newsgroups* dataset is fetched automatically via `scikit-learn`.


## 👤 Author

* **Yanghonghui Chen**
* **Guanhua Chen**