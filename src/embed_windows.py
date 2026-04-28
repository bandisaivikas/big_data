import os
import numpy as np
import pickle
from windower import create_windows
from embedder import embed_texts

def embed_all_windows(parquet_path: str, output_dir: str = "data/embeddings"):
    os.makedirs(output_dir, exist_ok=True)
    windows = create_windows(parquet_path)
    
    print(f"\nEmbedding {len(windows)} windows...")
    embedded = []
    
    for i, w in enumerate(windows):
        print(f"  Window {i+1}/{len(windows)}: {w['start'].date()} to {w['end'].date()} ({len(w['texts'])} articles)", end=" ")
        embeddings = embed_texts(w["texts"])
        
        record = {
            "start": w["start"],
            "end": w["end"],
            "embeddings": embeddings
        }
        embedded.append(record)
        
        path = os.path.join(output_dir, f"window_{i:03d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(record, f)
        print(f"→ shape {embeddings.shape} saved")
    
    print(f"\nDone. {len(embedded)} window embeddings saved to {output_dir}/")
    return embedded

if __name__ == "__main__":
    embed_all_windows("data/raw/corpus.parquet")
