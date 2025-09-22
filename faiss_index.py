import os
import faiss  # pip install faiss-cpu
import numpy as np
from typing import Tuple, List
from db_utils import fetch_all_vectors


INDEX_FILENAME = "faiss.index"


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    return (mat / norms).astype(np.float32)


def _default_index_path(index_path: str = None) -> str:
    if index_path:
        return index_path
    temp_dir = os.environ.get("TEMP") or os.environ.get("TMP") or os.getcwd()
    return os.path.join(temp_dir, INDEX_FILENAME)


def build_index(db_path: str = None, index_path: str = None) -> Tuple[str, int, int]:
    """Build a FAISS index from all x-vectors in DB. Returns (index_path, n_vectors, dim)."""
    vectors, ids = fetch_all_vectors(db_path)
    if vectors.size == 0:
        raise RuntimeError("DB에 x-vector가 없습니다. 먼저 학습을 수행하세요.")
    vectors = l2_normalize(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine 유사도 == inner product after L2 norm
    # IndexIDMap은 비어있는 인덱스만 허용 → base에 add하지 말고, IDMap에 add_with_ids만 수행
    id_index = faiss.IndexIDMap(index)
    id_index.add_with_ids(vectors, np.array(ids, dtype=np.int64))

    path = _default_index_path(index_path)
    faiss.write_index(id_index, path)
    return path, len(ids), dim


def load_index(index_path: str = None) -> faiss.Index:
    path = _default_index_path(index_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다: {path}")
    return faiss.read_index(path)


def search_top_k(query: np.ndarray, k: int = 5, index_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, ids) for top-k. query shape: [D] or [1, D]."""
    if query.ndim == 1:
        query = query.reshape(1, -1)
    query = l2_normalize(query)
    index = load_index(index_path)
    scores, ids = index.search(query.astype(np.float32), k)
    return scores[0], ids[0]


