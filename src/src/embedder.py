from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: list[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False)

if __name__ == "__main__":
    sample = ["AI is transforming healthcare", "Stock markets fell today"]
    embeddings = embed_texts(sample)
    print(f"Shape: {embeddings.shape}")
    print("Embeddings working correctly.")