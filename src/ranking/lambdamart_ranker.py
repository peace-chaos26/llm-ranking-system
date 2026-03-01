# src/ranking/lambdamart_ranker.py
"""
Stage 2: LambdaMART Learning-to-Rank model.

LambdaMART optimizes NDCG directly using gradient boosting.
Unlike pointwise models (predict relevance score per doc),
LambdaMART is a listwise model — it learns from the relative
ordering of documents within a query's result list.

How LambdaMART works:
1. For each query, you have a ranked list of (passage, relevance_label) pairs
2. LambdaMART computes "lambdas" — gradients that represent how much swapping two documents would change NDCG
3. Gradient boosted trees are trained to predict these lambdas
4. At inference: score each (query, passage) pair → sort by score

Why this beats a simple relevance classifier:
A classifier trained on binary relevance (relevant/not) optimizes accuracy.
LambdaMART optimizes the actual ranking metric (NDCG).
The distinction matters: getting rank 1 right is more important than
getting rank 50 right — LambdaMART knows this, a classifier doesn't.
"""

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
from loguru import logger

from src.ranking.features import compute_features, RankingFeatures


class LambdaMARTRanker:

    def __init__(self, top_k_output: int = 20):
        self.top_k_output = top_k_output
        self.model = None
        self._trained = False

    def prepare_training_data(
        self,
        queries: dict[str, str],
        corpus: dict[str, str],
        retrieval_results: dict[str, list[tuple[str, float]]],
        qrels: dict[str, dict[str, int]],
        bm25_scores: dict[str, dict[str, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build feature matrix for LambdaMART training.
        
        LambdaMART needs:
        - X: feature matrix (n_samples, n_features)
        - y: relevance labels (n_samples,)
        - groups: number of docs per query (for listwise training)
        """
        X, y, groups = [], [], []

        for qid, query_text in queries.items():
            if qid not in retrieval_results or qid not in qrels:
                continue

            candidates = retrieval_results[qid]
            relevant = qrels[qid]
            group_size = 0

            for rank, (pid, dense_score) in enumerate(candidates, start=1):
                if pid not in corpus:
                    continue

                passage_text = corpus[pid]
                bm25_score = (bm25_scores or {}).get(qid, {}).get(pid, 0.0)

                features = compute_features(
                    query=query_text,
                    passage=passage_text,
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    retrieval_rank=rank,
                )

                X.append(features.to_array())
                y.append(relevant.get(pid, 0))
                group_size += 1

            if group_size > 0:
                groups.append(group_size)

        return np.array(X), np.array(y), np.array(groups)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        groups_val: np.ndarray | None = None,
    ) -> None:
        """
        Train LambdaMART using LightGBM's lambdarank objective.
        
        Key hyperparameters:
        - num_leaves: complexity of each tree (higher = more expressive, overfits faster)
        - learning_rate: step size for gradient updates
        - n_estimators: number of trees (more = better up to a point)
        - min_child_samples: regularization — min docs per leaf
        """
        logger.info(f"Training LambdaMART on {len(groups_train)} queries, {len(X_train)} pairs...")

        train_data = lgb.Dataset(
            X_train, label=y_train,
            group=groups_train,
            feature_name=RankingFeatures.feature_names(),
        )

        callbacks = [lgb.log_evaluation(period=20)]
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None:
            val_data = lgb.Dataset(
                X_val, label=y_val,
                group=groups_val,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(stopping_rounds=10))

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 5,
            "importance_type": "gain",
            "verbosity": -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self._trained = True
        logger.info("LambdaMART training complete.")
        self._log_feature_importance()

    def _log_feature_importance(self) -> None:
        importance = self.model.feature_importance(importance_type="gain")
        names = RankingFeatures.feature_names()
        ranked = sorted(zip(names, importance), key=lambda x: x[1], reverse=True)
        logger.info("Feature importance (gain):")
        for name, score in ranked:
            logger.info(f"  {name:<30} {score:.1f}")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: dict[str, str],
        bm25_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Re-rank candidates using LambdaMART scores.
        Returns top_k_output candidates sorted by model score.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        features = []
        valid_candidates = []

        for rank, (pid, dense_score) in enumerate(candidates, start=1):
            if pid not in corpus:
                continue
            bm25_score = (bm25_scores or {}).get(pid, 0.0)
            feat = compute_features(
                query=query,
                passage=corpus[pid],
                dense_score=dense_score,
                bm25_score=bm25_score,
                retrieval_rank=rank,
            )
            features.append(feat.to_array())
            valid_candidates.append((pid, dense_score))

        if not features:
            return candidates[:self.top_k_output]

        X = np.array(features)
        scores = self.model.predict(X)

        ranked = sorted(
            zip(valid_candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(pid, float(score)) for (pid, _), score in ranked[:self.top_k_output]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"LambdaMART model saved to {path}")

    def load(self, path: str | Path) -> None:
        self.model = lgb.Booster(model_file=str(path))
        self._trained = True
        logger.info(f"LambdaMART model loaded from {path}")