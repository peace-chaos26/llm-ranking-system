# Multi-Stage LLM-Augmented Ranking System

A production-grade, domain-agnostic ranking pipeline implementing candidate retrieval,
multi-stage ML + LLM re-ranking, diversity post-processing, rigorous evaluation,
and feedback simulation with position bias modeling.

Built as a portfolio project to demonstrate end-to-end ML engineering — from data
pipelines and ANN indexing to LLM integration, offline evaluation, and feedback loop design.

---

## System Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │                   RANKING PIPELINE                  │
                        └─────────────────────────────────────────────────────┘

 Query
   │
   ▼
┌──────────────────────────────────────┐
│  STAGE 1: Retrieval                  │   Recall@100 = 0.908
│                                      │   Latency: ~440ms (batch)
│  ┌─────────────┐  ┌───────────────┐  │
│  │ Dense FAISS │  │  BM25 Sparse  │  │
│  │ IVF Index   │  │  (rank-bm25)  │  │
│  │ MiniLM-L6   │  │               │  │
│  └──────┬──────┘  └───────┬───────┘  │
│         │    RRF Fusion   │          │
│         └────────┬────────┘          │
│                  │  top-100          │
└──────────────────┼───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  STAGE 2: LambdaMART Ranker          │   NDCG@10: 0.367 → 0.435 (+18.3%)
│                                      │   Latency: +1ms
│  12 hand-crafted features            │   Cost: $0.00
│  Trained on hard negatives           │
│  Optimizes NDCG directly             │
│         top-100 → top-20             │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  STAGE 3: LLM Re-ranker              │   NDCG@10: 0.435 → 0.495 (+13.8%)
│                                      │   Latency: +23,000ms
│  GPT-4o-mini pointwise scoring       │   Cost: $0.0531/1k requests
│  Truncated to 200 words              │
│         top-20 → top-10              │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  POST-PROCESSING: MMR Diversity      │   Diversity tradeoff: λ=0.5
│                                      │   Latency: +few ms
│  Maximal Marginal Relevance          │   Cost: $0.00
│  λ balances relevance vs novelty     │
│         top-10 → final output        │
└──────────────────────────────────────┘
```

---

## Ablation Study Results

Evaluated on MS MARCO passage ranking dataset (100k corpus, 20 queries with relevance labels).

| Configuration | NDCG@10 | MRR@10 | R@10 | Latency | Cost/1k |
|---|---|---|---|---|---|
| 1. Retrieval only (Hybrid RRF) | 0.3674 | 0.2922 | 0.60 | 440ms | $0.00 |
| 2. + LambdaMART (Stage 2) | 0.4347 | 0.3713 | 0.65 | 441ms | $0.00 |
| 3. + LLM Re-ranker (Stage 3) | 0.4949 | 0.4283 | 0.70 | 23,930ms | $0.053 |
| 4. + MMR Diversity (Full) | 0.4949 | 0.4283 | 0.70 | 23,930ms | $0.053 |

**Key findings:**
- LambdaMART adds **+18.3% NDCG** at zero cost and <1ms latency — highest ROI component
- LLM re-ranking adds another **+13.8% NDCG** at $0.053/1k requests — clear cost/quality tradeoff
- MMR diversity does not affect NDCG on MS MARCO (binary relevance, 1 relevant doc/query) but improves intra-list variety for user experience

---

## Retrieval Benchmark

| Retriever | Recall@10 | Recall@20 | Recall@50 | Recall@100 | Latency |
|---|---|---|---|---|---|
| BM25 | 0.324 | 0.402 | 0.484 | 0.572 | 118ms |
| Dense (FAISS IVF) | 0.680 | 0.764 | 0.864 | 0.896 | 1.4ms |
| **Hybrid RRF** | **0.628** | **0.766** | **0.862** | **0.908** | 121ms |

Dense retrieval dominates BM25 on MS MARCO's short semantic queries (+56.6% Recall@100).
Hybrid RRF adds a further +1.3% at negligible latency cost. Hybrid is used as the retrieval
layer because even marginal recall gains at Stage 1 compound through the entire pipeline.

---

## Feedback Simulation

Simulated 100 user sessions per query across 30 queries with position bias (η=1.0).

**Position bias effect (click-through rate by rank):**
```
Rank  1: ~72% CTR  ████████████████████████████████████████
Rank  2: ~45% CTR  ██████████████████████
Rank  5: ~16% CTR  ████████
Rank 10: ~8%  CTR  ████
```

**Feature importance shift — the key finding:**

| Feature | Ground Truth Ranker | Click-Trained (Biased) |
|---|---|---|
| `dense_score` | **#1** (importance: 9) | #7 (importance: 52) |
| `retrieval_rank` | #5 (importance: 2) | **#1** (importance: 3006) |

Training on biased clicks caused `retrieval_rank` to dominate with 3006x higher importance
than in the ground-truth ranker. The model learned to optimize for **position** rather than
**relevance** — the core failure mode of naive feedback loops.

IPS (Inverse Propensity Scoring) debiasing addresses this by weighting clicks at lower
positions more heavily. At production scale (10k+ queries), IPS recovery is statistically
measurable. On this small dataset, the bucketing of continuous IPS scores into integer
labels collapsed the signal difference — a known limitation documented in the project.

---

## Design Decisions

### Why multi-stage ranking?

Running an LLM on 100 candidates per query at scale:
- 100 candidates × 500 tokens = 50,000 tokens per query
- At $0.00015/1k = $0.0075/query → $7,500/day at 1M queries

After LambdaMART reduces to 20 candidates, LLM cost drops 5x to $1,500/day.
Same quality. Multi-stage exists to apply expensive models only where they add value.

### Why FAISS over Qdrant?

FAISS runs locally with no infrastructure, making it ideal for development and benchmarking.
Qdrant is preferred in production when you need metadata filtering, real-time index updates,
or horizontal scaling. The retriever interface is designed to be swappable — replacing FAISS
with Qdrant requires only implementing `BaseRetriever`.

### Why LambdaMART over a neural cross-encoder at Stage 2?

| | LambdaMART | Cross-Encoder |
|---|---|---|
| Inference latency | <1ms per query | 50-100ms per query |
| Hardware | CPU | GPU preferred |
| Interpretability | Feature importance | Black box |
| Training data | ~2k queries sufficient | Needs 10k+ |

At 100 candidates per query, a cross-encoder at Stage 2 adds 5-10 seconds of latency.
LambdaMART adds <1ms. The quality delta doesn't justify the latency cost at this stage.

### Why sentence-transformers over OpenAI embeddings for retrieval?

Embedding 100k passages once costs ~$0.10 with OpenAI ada-002. Fine for production.
During development you rebuild the index many times. Sentence-transformers (MiniLM-L6-v2)
are free, run locally, and produce 384-dim vectors fast on CPU. The quality tradeoff is
acceptable for a retrieval recall problem — we just need good candidates, not perfect ranking.

### Why Reciprocal Rank Fusion over learned fusion?

RRF uses only rank positions — no score normalization needed across different retrieval systems.
Learned fusion requires training data and a trained model. RRF is parameter-free, deterministic,
and empirically competitive with learned fusion. It's only worth learning fusion weights when
you have click data to train on, which requires a live system.

### Why BM25 as a baseline?

On short keyword queries (avg 5.9 words in MS MARCO), BM25 is a surprisingly competitive
baseline. Any retrieval system that doesn't beat BM25 has a fundamental problem.
BM25 also catches exact product codes, names, and rare terms that dense models can miss.

---

## What This Teaches

| Concept | Where It Appears |
|---|---|
| ANN indexing (FAISS IVF, HNSW) | `src/retrieval/dense_retriever.py` |
| Hybrid search + RRF fusion | `src/retrieval/hybrid_retriever.py` |
| Learning-to-Rank (LambdaMART) | `src/ranking/lambdamart_ranker.py` |
| Hard negative mining | `scripts/train_ranker.py` |
| LLM integration + cost tracking | `src/ranking/llm_reranker.py` |
| MMR diversity | `src/ranking/diversity.py` |
| NDCG, MRR, Recall implementation | `src/evaluation/metrics.py` |
| Ablation study design | `scripts/run_evaluation.py` |
| Position bias modeling | `src/feedback/position_bias.py` |
| IPS debiasing | `src/feedback/feedback_processor.py` |
| Config-driven experiments | `configs/base.yaml` |
| Structured logging + cost reporting | `src/utils/` |

---

## Repo Structure

```
ranking-system/
│
├── src/
│   ├── retrieval/          # BM25, Dense FAISS, Hybrid RRF retrievers
│   ├── ranking/            # LambdaMART, LLM re-ranker, MMR diversity, pipeline
│   ├── evaluation/         # NDCG, MRR, Recall, diversity, coverage metrics
│   ├── feedback/           # Position bias, click simulator, IPS debiasing
│   └── utils/              # Config loader, structured logger, cost tracker
│
├── configs/
│   └── base.yaml           # All pipeline parameters — no hardcoding
│
├── scripts/
│   ├── download_data.py    # MS MARCO download + preprocessing
│   ├── validate_data.py    # Corpus integrity checks
│   ├── build_indexes.py    # FAISS + BM25 index construction
│   ├── run_retrieval.py    # Retrieval benchmark (Recall@k + latency)
│   ├── train_ranker.py     # LambdaMART training with hard negatives
│   ├── run_evaluation.py   # Full ablation study
│   └── run_feedback.py     # Feedback simulation + retraining comparison
│
├── experiments/
│   ├── models/             # Trained model checkpoints
│   └── results/            # JSON evaluation outputs (reproducible)
│
├── data/                   # Not committed — see setup instructions
├── .env.example            # API key template
└── pyproject.toml          # Dependencies + build config
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/ranking-system.git
cd ranking-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
brew install libomp          # macOS only — required for LightGBM
```

### 2. Set up API keys

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 3. Download data and build indexes

```bash
python -m scripts.download_data      # ~10 min, downloads MS MARCO
python -m scripts.validate_data      # sanity checks
python -m scripts.build_indexes      # ~10 min, builds FAISS + BM25 indexes
```

### 4. Run retrieval benchmark

```bash
python -m scripts.run_retrieval
```

### 5. Train the ranker

```bash
python -m scripts.train_ranker
```

### 6. Run full ablation evaluation

```bash
python -m scripts.run_evaluation
```

### 7. Run feedback simulation

```bash
python -m scripts.run_feedback
```

---

## Dataset

**MS MARCO Passage Ranking** (Microsoft Machine Reading Comprehension)
- 100,000 passages from the full 8.8M passage corpus
- 2,066 queries with human relevance judgments (qrels)
- Average query length: 5.9 words
- Average relevant passages per query: 1.01 (sparse relevance)
- Stored in TREC qrels format for compatibility with standard IR evaluation tools

---

## Environment

- Python 3.11
- macOS (Apple Silicon) — set `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1`
  to avoid segfaults from FAISS + HuggingFace thread conflicts
- All experiments run on CPU — no GPU required
- LLM evaluation requires OpenAI API key (GPT-4o-mini, ~$0.01 for 10 queries)