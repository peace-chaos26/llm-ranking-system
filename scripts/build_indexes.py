# scripts/build_indexes.py
"""
Build and save BM25 and FAISS indexes.
Run once — then load from disk for all experiments.

Why save indexes?
BM25 index: ~2 minutes to build
FAISS IVF index: ~5-10 minutes to build + encode 100k passages
"""

from pathlib import Path
from src.utils.data_loader import load_corpus_as_dict
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from loguru import logger

INDEX_DIR = Path("data/indexes")
CORPUS_PATH = "data/processed/corpus.jsonl"


def main():
    corpus = load_corpus_as_dict(CORPUS_PATH)

    # --- BM25 ---
    logger.info("=== Building BM25 Index ===")
    bm25 = BM25Retriever()
    bm25.index(corpus)
    bm25.save(INDEX_DIR / "bm25.pkl")

    # --- Dense (FAISS IVF) ---
    logger.info("=== Building Dense Index ===")
    dense = DenseRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="IVF",
        nlist=256,
        nprobe=32,
    )
    dense.index(corpus)
    dense.save(INDEX_DIR / "faiss_ivf.index", INDEX_DIR / "passage_ids.json")

    logger.info("=== All indexes built and saved ===")


if __name__ == "__main__":
    main()