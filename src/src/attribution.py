import pickle
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from drift import compute_wkcs

def get_top_words(topic_model, topic_id, n=5):
    words = topic_model.get_topic(topic_id)
    if words:
        return ", ".join([w[0] for w in words[:n]])
    return "unknown"

def attribute_drift(window1: dict, window2: dict, n_topics: int = 8) -> dict:
    all_texts = window1["texts"] + window2["texts"]
    
    vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1,2))
    topic_model = BERTopic(vectorizer_model=vectorizer, nr_topics=n_topics, verbose=False)
    topics, _ = topic_model.fit_transform(all_texts)
    
    w1_topics = topics[:len(window1["texts"])]
    w2_topics = topics[len(window1["texts"]):]
    
    unique_topics = [t for t in set(topics) if t != -1]
    
    topic_drift_scores = []
    for topic_id in unique_topics:
        w1_idx = [i for i, t in enumerate(w1_topics) if t == topic_id]
        w2_idx = [i for i, t in enumerate(w2_topics) if t == topic_id]
        
        if len(w1_idx) < 5 or len(w2_idx) < 5:
            continue
        
        e1 = window1["embeddings"][w1_idx]
        e2 = window2["embeddings"][w2_idx]
        
        scores = compute_wkcs(e1, e2)
        topic_words = get_top_words(topic_model, topic_id)
        
        topic_drift_scores.append({
            "topic_id": topic_id,
            "topic_words": topic_words,
            "wkcs": scores["wkcs"],
            "wasserstein": scores["wasserstein"],
            "kl_divergence": scores["kl_divergence"],
            "w1_articles": len(w1_idx),
            "w2_articles": len(w2_idx)
        })
    
    if not topic_drift_scores:
        return {"topics": [], "top_driver": "insufficient data"}
    
    df = pd.DataFrame(topic_drift_scores).sort_values("wkcs", ascending=False)
    total = df["wkcs"].sum()
    df["drift_contribution_pct"] = (df["wkcs"] / total * 100).round(1)
    
    return {
        "topics": df.to_dict("records"),
        "top_driver": df.iloc[0]["topic_words"]
    }

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from windower import create_windows
    
    windows = create_windows("data/raw/corpus.parquet")
    emb_files = sorted(os.listdir("data/embeddings"))
    
    with open(f"data/embeddings/{emb_files[13]}", "rb") as f:
        w1_emb = pickle.load(f)
    with open(f"data/embeddings/{emb_files[14]}", "rb") as f:
        w2_emb = pickle.load(f)
    
    w1 = {"texts": windows[13]["texts"], "embeddings": w1_emb["embeddings"]}
    w2 = {"texts": windows[14]["texts"], "embeddings": w2_emb["embeddings"]}
    
    print("Analyzing Pair 14 (highest drift: Sep 28 → Oct 28, 2017)")
    print("=" * 60)
    print("Running BERTopic attribution...")
    result = attribute_drift(w1, w2)
    
    print(f"\nTop drift-driving topics:\n")
    for i, t in enumerate(result["topics"][:5]):
        print(f"  #{i+1} [{t['drift_contribution_pct']}% of drift]")
        print(f"      Keywords : {t['topic_words']}")
        print(f"      WKCS     : {t['wkcs']:.4f}")
        print(f"      Articles : {t['w1_articles']} → {t['w2_articles']}")
        print()
    
    print(f"Primary drift driver: {result['top_driver']}")
