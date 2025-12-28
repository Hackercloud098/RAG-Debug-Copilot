import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
OUT_DIR = ROOT / "data" / "processed"
INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH = OUT_DIR / "faiss_meta.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Run prepare_logs.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]
    print(f"Loaded chunks: {len(texts)}")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(INDEX_PATH))

    with META_PATH.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            meta = {
                "row_id": i,
                "chunk_id": c.get("chunk_id"),
                "log_id": c.get("log_id"),
                "source": c.get("source"),
                "tool": c.get("tool"),
                "tags": c.get("tags", []),
                "timestamp": c.get("timestamp"),
                "text": c.get("text"),
                "provenance": c.get("provenance", None),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"Wrote FAISS index: {INDEX_PATH}")
    print(f"Wrote metadata:   {META_PATH}")

if __name__ == "__main__":
    main()
