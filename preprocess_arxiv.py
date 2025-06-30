import json
import random
import pandas as pd
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

SAMPLE_SIZE = 5000

def load_sample_arxiv(file_path, sample_size):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:  # Added encoding
        for line in tqdm(f, desc="Reading arXiv data"):
            if random.random() < sample_size/2_000_000:
                try:
                    entry = json.loads(line)
                    # Extract year safely
                    journal_ref = entry.get("journal_ref", "")
                    year = int(journal_ref[-4:]) if journal_ref and journal_ref[-4:].isdigit() else None
                    
                    data.append({
                        "id": entry["id"],
                        "title": entry["title"],
                        "abstract": entry["abstract"],
                        "categories": entry["categories"].split(),
                        "year": year
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping malformed entry: {e}")
                
                if len(data) >= sample_size:
                    break
    return pd.DataFrame(data)

# Fixed path using raw string
ARXIV_FILE = r"C:\Users\masou\.cache\kagglehub\datasets\Cornell-University\arxiv\versions\238\arxiv-metadata-oai-snapshot.json"

arxiv_df = load_sample_arxiv(ARXIV_FILE, SAMPLE_SIZE)

arxiv_df.to_csv("arxiv_sample.csv", index=False)
print(f"Saved {len(arxiv_df)} papers to arxiv_sample.csv")

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab',   quiet=True)

def preprocess_text(text):
    """Clean and normalize text for clustering"""
    # 1. Remove LaTeX equations
    text = re.sub(r'\$.*?\$', '', text)
    
    # 2. Remove special characters/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Lowercase and tokenize
    words = nltk.word_tokenize(text.lower())
    
    # 4. Remove stopwords and short tokens
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(w) for w in words)


arxiv_df["processed_text"] = arxiv_df["title"] + " " + arxiv_df["abstract"]
arxiv_df["processed_text"] = arxiv_df["processed_text"].apply(preprocess_text)

# Save preprocessed data
arxiv_df.to_csv("preprocessed_arxiv.csv", index=False)