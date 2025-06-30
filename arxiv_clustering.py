import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['USE_TORCH'] = '1'  # Force transformers to use PyTorch
import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, fowlkes_mallows_score
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from dotenv import load_dotenv
from groq import Groq  #use free groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# ========================
# CONFIGURATION
# ========================
class Config:
    SAMPLE_SIZE = 5000                   # Subsample for stability trials
    EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
    UMAP_COMPONENTS = 32                 # Reduced dimensions
    MAX_CLUSTERS = 15                    # Max k for optimization
    STABILITY_TRIALS = 30                # Trials for stability metric
    LLM_LABELING = True                 # Enable GPT cluster labeling
    BATCH_SIZE = 128                     # For embedding generation
    TECHNICAL_TERMS = {                  # Domain glossary
        'transformer', 'llm', 'gan', 'bert', 'rnn', 'crispr', 'quantum', 
        'granger', 'topology', 'eigenvalue', 'transformer', 'translation',
        'variational', 'convolutional', 'generative', 'diffusion'
    }
    ABBREVIATION_MAP = {                 # For text expansion
        'ai': 'artificial intelligence',
        'ml': 'machine learning',
        'nlp': 'natural language processing',
        'cv': 'computer vision',
        'gan': 'generative adversarial network',
        'bert': 'bidirectional encoder representations from transformers'
    }

# ========================
# CORE FUNCTIONS
# ========================
def enhance_domain_terms(text: str) -> str:
    """Expand abbreviations and highlight technical terms"""
    # Expand abbreviations
    for abbr, full in Config.ABBREVIATION_MAP.items():
        text = re.sub(rf'\b{abbr}\b', full, text, flags=re.IGNORECASE)
    
    # Highlight technical terms
    words = text.split()
    enhanced = [w.upper() if w.lower() in Config.TECHNICAL_TERMS else w for w in words]
    return " ".join(enhanced)

def get_embeddings(texts: list, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings with batch processing"""
    return model.encode(
        texts, 
        batch_size=Config.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    ).cpu().numpy()

def reduce_dimensionality(embeddings: np.ndarray) -> np.ndarray:
    """Apply UMAP for dimensionality reduction"""
    return UMAP(
        n_components=Config.UMAP_COMPONENTS,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        metric='cosine'
    ).fit_transform(embeddings)

def cluster_stability(embeddings: np.ndarray, k: int) -> float:
    """Compute cluster stability using Fowlkes-Mallows score"""
    base_labels = AgglomerativeClustering(
        n_clusters=k, 
        metric='cosine', 
        linkage='average'
    ).fit_predict(embeddings)
    
    scores = []
    for _ in range(Config.STABILITY_TRIALS):
        # Create random subset
        subset_idx = np.random.choice(
            len(embeddings), 
            size=int(0.8*len(embeddings)),
            replace=False
        )
        subset_emb = embeddings[subset_idx]
        
        # Cluster subset
        subset_labels = AgglomerativeClustering(
            n_clusters=k, 
            metric='cosine', 
            linkage='average'
        ).fit_predict(subset_emb)
        
        # Compare with base clustering
        score = fowlkes_mallows_score(
            base_labels[subset_idx], 
            subset_labels
        )
        scores.append(score)
        
    return np.mean(scores)

def find_optimal_clusters(embeddings: np.ndarray) -> int:
    """Determine optimal k using silhouette score and stability"""
    silhouette_scores = []
    stability_scores = []
    k_range = range(2, Config.MAX_CLUSTERS+1)
    
    for k in tqdm(k_range, desc="Evaluating cluster counts"):
        # Silhouette score
        labels = AgglomerativeClustering(
            n_clusters=k, 
            metric='cosine', 
            linkage='average'
        ).fit_predict(embeddings)
        sil_score = silhouette_score(embeddings, labels, metric='cosine')
        silhouette_scores.append(sil_score)
        
        # Stability score
        if k <= 10:  # Only compute for smaller k (computationally expensive)
            stab_score = cluster_stability(embeddings, k)
            stability_scores.append(stab_score)
    
    # Weighted score (70% silhouette, 30% stability)
    combined_scores = []
    for i, sil in enumerate(silhouette_scores):
        if i < len(stability_scores):
            combined = 0.7*sil + 0.3*stability_scores[i]
        else:
            combined = sil
        combined_scores.append(combined)
    
    optimal_k = k_range[np.argmax(combined_scores)]
    print(f"Optimal clusters: k={optimal_k} (Silhouette: {max(silhouette_scores):.3f})")
    return optimal_k

def generate_cluster_label(sentences: list, cluster_id: int) -> str:
    """Generate labels using KeyBERT + Groq refinement"""
    # 1. KeyBERT labeling - always have a valid fallback
    kw_model = KeyBERT()
    cluster_doc = " ".join(sentences)
    keywords = kw_model.extract_keywords(
        cluster_doc,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=5,
        diversity=0.7
    )
    keybert_label = " & ".join([kw[0] for kw in keywords[:2]]).title()
    
    # 2. Groq refinement (if enabled)
    if Config.LLM_LABELING and os.getenv("GROQ_API_KEY"):
        # Improved prompt with strict constraints
        prompt = (
            "Generate exactly ONE 2-5 word academic topic label for these research abstracts. "
            "Return ONLY the label itself with NO additional text, explanations, or formatting. "
            "DO NOT include numbers, quotes, or special characters.\n\n"
            "Examples of valid responses:\n"
            "Quantum Computing & AI\n"
            "Transformer Architectures\n"
            "Genomic Sequencing Techniques\n\n"
            "Abstracts:\n" + "\n".join(sentences[:3]) + 
            f"\n\nCurrent label: '{keybert_label}'. Improved label:"
        )
        
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an academic domain expert. Return ONLY the topic label with no other text."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=15,
                temperature=0.1,
                stop=["\n", ".", ":", "Label"]  # Stop at common sentence endings
            )
            
            # Extract and clean the label
            raw_label = response.choices[0].message.content.strip()
            
            # Clean using multiple strategies
            clean_label = clean_llm_output(raw_label, keybert_label)
            
            return f"{clean_label} (LLM-enhanced)"
                
        except Exception as e:
            print(f"Groq labeling failed: {e}")
    
    return keybert_label

def clean_llm_output(raw_label: str, fallback: str) -> str:
    """Robust cleaning of LLM output with multiple fallbacks"""
    # Strategy 1: Remove known problematic prefixes
    prefixes = [
        "here are", "topic labels", "label:", "suggestion:", 
        "based on", "the label is", "improved label"
    ]
    for prefix in prefixes:
        if raw_label.lower().startswith(prefix):
            raw_label = raw_label[len(prefix):].strip(" :.-")
    
    # Strategy 2: Extract first valid phrase
    words = raw_label.split()
    valid_phrases = []
    current_phrase = []
    
    for word in words:
        # Remove special characters
        clean_word = re.sub(r'[^a-zA-Z&]', '', word)
        if not clean_word:
            continue
            
        # Start new phrase after stop words
        if clean_word.lower() in {"and", "&", "or", "vs"} and current_phrase:
            valid_phrases.append(" ".join(current_phrase))
            current_phrase = []
        else:
            current_phrase.append(clean_word)
            
        # Limit to 5 words
        if len(current_phrase) >= 5:
            valid_phrases.append(" ".join(current_phrase))
            current_phrase = []
    
    if current_phrase:
        valid_phrases.append(" ".join(current_phrase))
    
    # Strategy 3: Select best candidate
    for phrase in valid_phrases:
        if 2 <= len(phrase.split()) <= 5 and len(phrase) < 50:
            return phrase.title()
    
    # Strategy 4: Return first 5 words as fallback
    clean_words = [re.sub(r'[^a-zA-Z]', '', w) for w in words[:5] if w]
    clean_words = [w for w in clean_words if w and len(w) > 1]
    
    if 2 <= len(clean_words) <= 5:
        return " ".join(clean_words).title()
    
    # Final fallback to KeyBERT label
    return fallback
# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    # 1. Load data with domain enhancement
    df = pd.read_csv("preprocessed_arxiv.csv")
    df["enhanced_text"] = df["processed_text"].apply(enhance_domain_terms)
    
    # 2. Embedding generation
    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    embeddings = get_embeddings(df["enhanced_text"].tolist(), model)
    
    # 3. Dimensionality reduction
    reduced_emb = reduce_dimensionality(embeddings)
    
    # 4. Optimal clustering
    optimal_k = find_optimal_clusters(reduced_emb)
    
    # 5. Perform clustering
    clusterer = AgglomerativeClustering(
        n_clusters=optimal_k,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clusterer.fit_predict(reduced_emb)
    df["cluster"] = cluster_labels
    
    # 6. Generate cluster labels
    cluster_results = {}
    for cluster_id in tqdm(range(optimal_k), desc="Labeling clusters"):
        cluster_sents = df[df["cluster"] == cluster_id]["enhanced_text"].tolist()
        label = generate_cluster_label(cluster_sents, cluster_id)
        cluster_results[cluster_id] = label
    
    # 7. Save results
    df["cluster_label"] = df["cluster"].map(cluster_results)
    df.to_csv("clustered_arxiv.csv", index=False)
    
    # 8. Print summary
    print("\n=== CLUSTERING SUMMARY ===")
    print(f"Total papers: {len(df)}")
    print(f"Clusters: {optimal_k}")
    for cid, label in cluster_results.items():
        size = sum(df["cluster"] == cid)
        print(f"Cluster {cid} ({size} papers): {label}")