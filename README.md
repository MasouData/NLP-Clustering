# Advanced NLP Clustering for Academic Papers

This project implements an advanced NLP pipeline for clustering academic papers using state-of-the-art techniques:
- Sentence Transformers for semantic embeddings
- UMAP for dimensionality reduction
- Agglomerative Clustering with stability validation
- Hybrid labeling with KeyBERT and Groq's LLM (Llama 3)

## Features ✨
- **Domain-Specific Enhancement**: Technical term highlighting and abbreviation expansion
- **Optimal Cluster Detection**: Combined silhouette score and stability metric
- **Advanced Labeling**: Hybrid KeyBERT + LLM approach with robust output cleaning
- **Efficient Processing**: Batch embedding generation and UMAP dimensionality reduction
- **Reproducible Research**: Full configuration management

## Installation 🛠️
1. Clone repository:
```bash
git clone https://github.com/YOUR_USERNAME/nlp-clustering.git
cd nlp-clustering

2. Install dependencies:
```bash
pip install -r requirements.txt

3. Set up environment variables:
```bash
echo "GROQ_API_KEY=your_api_key_here" > .env


## Usage 🚀
Step 1: Download Data
- Get arXiv dataset from Kaggle(https://www.kaggle.com/datasets/Cornell-University/arxiv/data)
- Place arxiv-metadata-oai-snapshot.json in the same directory of py file.

Step 2: Preprocess Data
```bash 
python preprocess_arxiv.py

Step 3: Run Clustering Pipeline
```bash
python arxiv_clustering.py


## Workflow Overview 🔄
graph TD
    A[Raw arXiv Data] --> B(Preprocessing)
    B --> C[Embedding Generation]
    C --> D[UMAP Dimensionality Reduction]
    D --> E[Cluster Optimization]
    E --> F[Semantic Labeling]
    F --> G[Results Analysis]

## Analyze Results (Example)
```bash
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("clustered_arxiv.csv")
print(f"Total papers: {len(df)}")
print(f"Cluster distribution:\n{df['cluster'].value_counts()}")
df['cluster_label'].value_counts().plot.barh()
plt.title("Paper Distribution by Topic Cluster")
plt.tight_layout()
plt.savefig("cluster_distribution.png")
