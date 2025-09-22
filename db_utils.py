import os
import sqlite3
import numpy as np
from typing import List, Tuple, Optional


DB_FILENAME = "xvectors.sqlite3"


def get_db_path(db_path: Optional[str] = None) -> str:
    return db_path if db_path else os.path.join(os.getcwd(), DB_FILENAME)


def init_db(db_path: Optional[str] = None) -> None:
    """Create the SQLite schema if it does not exist."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS xvectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                singer TEXT NOT NULL,
                file_path TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(singer, file_path)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _to_blob(vector: np.ndarray) -> bytes:
    if vector.dtype != np.float32:
        vector = vector.astype(np.float32)
    return vector.tobytes()


def upsert_xvector(singer: str, file_path: str, vector: np.ndarray, db_path: Optional[str] = None) -> int:
    """Insert or replace x-vector for a singer+file. Returns row id."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        dim = int(vector.shape[0])
        cur.execute(
            """
            INSERT INTO xvectors (singer, file_path, dim, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(singer, file_path) DO UPDATE SET
                dim=excluded.dim,
                vector=excluded.vector,
                created_at=CURRENT_TIMESTAMP
            """,
            (singer, file_path, dim, _to_blob(vector)),
        )
        conn.commit()
        return cur.lastrowid if cur.lastrowid is not None else cur.execute("SELECT id FROM xvectors WHERE singer=? AND file_path=?", (singer, file_path)).fetchone()[0]
    finally:
        conn.close()


def fetch_all_vectors(db_path: Optional[str] = None) -> Tuple[np.ndarray, List[int]]:
    """Return matrix [N, D] and list of db ids [N]."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, dim, vector FROM xvectors")
        rows = cur.fetchall()
        ids: List[int] = []
        vectors: List[np.ndarray] = []
        for row in rows:
            row_id, dim, blob = row
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size != dim:
                continue
            ids.append(row_id)
            vectors.append(vec)
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32), []
        mat = np.stack(vectors).astype(np.float32)
        return mat, ids
    finally:
        conn.close()


def fetch_vectors_by_singer(singer: str, db_path: Optional[str] = None) -> np.ndarray:
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT dim, vector FROM xvectors WHERE singer=?", (singer,))
        rows = cur.fetchall()
        vectors: List[np.ndarray] = []
        for dim, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == dim:
                vectors.append(vec)
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(vectors).astype(np.float32)
    finally:
        conn.close()


def get_singer_by_id(row_id: int, db_path: Optional[str] = None) -> Optional[str]:
    path = get_db_path(db_path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT singer FROM xvectors WHERE id=?", (row_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


