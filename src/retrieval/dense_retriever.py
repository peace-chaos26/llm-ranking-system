# src/retrieval/dense_retriever.py
"""
Dense retriever using sentence-transformers embeddings + FAISS ANN index.

1. Encode every passage into a fixed-size vector (embedding) offline
2. Store all vectors in a FAISS index
3. At query time: encode query → find nearest vectors → return those passages

FAISS Index Types — the core tradeoff:
┌──────────┬──────────┬─────────┬──────────────────────────────────┐
│ Type     │ Recall   │ Speed   │ When to use                      │
├──────────┼──────────┼─────────┼──────────────────────────────────┤
│ Flat     │ 100%     │ Slow    │ < 10k vectors, need exact results │
│ IVF      │ ~95-98%  │ Fast    │ 100k-10M vectors, our case       │
│ HNSW     │ ~96-99%  │ Fastest │ Production, memory not a concern  │
└──────────┴──────────┴─────────┴──────────────────────────────────┘

We use IVF (Inverted File Index):
- Divides vector space into nlist clusters (Voronoi cells)
- At query time, only searches nprobe nearest clusters
- nprobe is the recall vs speed knob: higher = better recall, slower
"""

import numpy as np
import faiss
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "IVF",    # "Flat" | "IVF" | "HNSW"
        nlist: int = 256,           # IVF clusters — sqrt(corpus_size) is a good rule of thumb
        nprobe: int = 32,           # IVF clusters to search — higher = better recall, slower
        batch_size: int = 512,
    ):
        self.model_name = model_name
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.batch_size = batch_size

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        self.faiss_index = None
        self._indexed = False
        self.passage_ids: list[str] = []
        self._indexed = False

    def _encode(self, texts: list[str], desc: str = "Encoding") -> np.ndarray:
        """Encode texts to normalized embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalize — enables cosine sim via dot product
        )
        return embeddings.astype(np.float32)

    def _build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index from embeddings.
        
        After L2 normalization, inner product == cosine similarity.
        FAISS has highly optimized inner product search (BLAS).
        This is faster than using IndexFlatL2 on unnormalized vectors.
        """
        n = embeddings.shape[0]

        if self.index_type == "Flat":
            index = faiss.IndexFlatIP(self.dim)

        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Training IVF index with {self.nlist} clusters on {n:,} vectors...")
            index.train(embeddings)
            index.nprobe = self.nprobe

        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        return index

    def index(self, corpus: dict[str, str]) -> None:
        """Encode all passages and build FAISS index."""
        logger.info(f"Building {self.index_type} dense index over {len(corpus):,} passages...")

        self.passage_ids = list(corpus.keys())
        texts = [corpus[pid] for pid in self.passage_ids]

        embeddings = self._encode(texts, desc="Encoding corpus")

        self.faiss_index = self._build_index(embeddings)
        self.faiss_index.add(embeddings)
        self._indexed = True
        logger.info(f"Dense index built. Total vectors: {self.faiss_index.ntotal:,}")

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Retrieve top-k passages using ANN search."""
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve()")

        query_embedding = self._encode([query])
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:   # FAISS returns -1 for empty slots
                results.append((self.passage_ids[idx], float(score)))

        return results

    def batch_retrieve(
        self, queries: dict[str, str], top_k: int = 100
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Batch retrieval — encode all queries at once, then search.
        Much faster than calling retrieve() in a loop because the
        embedding model processes queries in parallel batches.
        """
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve()")

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        query_embeddings = self._encode(query_texts, desc="Encoding queries")
        scores_matrix, indices_matrix = self.faiss_index.search(query_embeddings, top_k)

        results = {}
        for qid, scores, indices in zip(query_ids, scores_matrix, indices_matrix):
            results[qid] = [
                (self.passage_ids[idx], float(score))
                for score, idx in zip(scores, indices)
                if idx != -1
            ]

        return results

    def save(self, index_path: str | Path, ids_path: str | Path) -> None:
        """Save FAISS index and passage ID mapping."""
        index_path, ids_path = Path(index_path), Path(ids_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss_index, str(index_path))

        import json
        with open(ids_path, "w") as f:
            json.dump(self.passage_ids, f)

        logger.info(f"Dense index saved to {index_path}")

    def load(self, index_path: str | Path, ids_path: str | Path) -> None:
        """Load saved FAISS index."""
        import json
        self.faiss_index = faiss.read_index(str(index_path))
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = self.nprobe

        with open(ids_path) as f:
            self.passage_ids = json.load(f)

        self._indexed = True
        logger.info(f"Dense index loaded. Vectors: {self.faiss_index.ntotal:,}")