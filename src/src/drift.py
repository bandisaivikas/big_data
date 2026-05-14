import numpy as np
from scipy.stats import entropy
import ot

def estimate_distribution(emb1: np.ndarray, emb2: np.ndarray, n_components: int = 50):
    """Project both windows to same dimensionality."""
    max_components = min(n_components, emb1.shape[0], emb2.shape[0], emb1.shape[1])
    
    def project(e, Vt):
        return e @ Vt[:max_components].T
    
    _, _, Vt1 = np.linalg.svd(emb1 - emb1.mean(0), full_matrices=False)
    _, _, Vt2 = np.linalg.svd(emb2 - emb2.mean(0), full_matrices=False)
    
    return project(emb1, Vt1), project(emb2, Vt2)

def compute_kl_divergence(emb1: np.ndarray, emb2: np.ndarray, bins: int = 50) -> float:
    e1, e2 = estimate_distribution(emb1, emb2)
    p1 = e1[:, 0]
    p2 = e2[:, 0]
    
    min_val = min(p1.min(), p2.min())
    max_val = max(p1.max(), p2.max())
    bins_range = np.linspace(min_val, max_val, bins)
    
    hist1, _ = np.histogram(p1, bins=bins_range, density=True)
    hist2, _ = np.histogram(p2, bins=bins_range, density=True)
    
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    
    return float(entropy(hist1, hist2))

def compute_wasserstein(emb1: np.ndarray, emb2: np.ndarray) -> float:
    e1, e2 = estimate_distribution(emb1, emb2)
    
    a = np.ones(len(e1)) / len(e1)
    b = np.ones(len(e2)) / len(e2)
    
    M = ot.dist(e1, e2, metric='sqeuclidean')
    M /= M.max()
    
    return float(ot.emd2(a, b, M))

def compute_wkcs(emb1: np.ndarray, emb2: np.ndarray, alpha: float = 0.6, beta: float = 0.4) -> dict:
    w2 = compute_wasserstein(emb1, emb2)
    kl = compute_kl_divergence(emb1, emb2)
    return {
        "wasserstein": w2,
        "kl_divergence": kl,
        "wkcs": alpha * w2 + beta * kl
    }
