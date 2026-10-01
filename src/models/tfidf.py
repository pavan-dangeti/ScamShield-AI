"""TF-IDF + logistic regression baseline, retrained on the licence-checked corpus.

This is the incumbent model, retrained under the new data pipeline so that the
comparison in ``scripts/evaluate_models.py`` is fair: same splits, same
evaluation, same interface as the neural models.
"""

from __future__ import annotations

import json
import os
import pickle
import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

from src.data import load_split
from src.models.base import Prediction, TacticPrediction
from src.taxonomy import TACTICS


class TfidfScamModel:
    """Word + character TF-IDF into one-vs-rest logistic regression heads."""

    name = "tfidf-lr"

    def __init__(self, c_grid=(0.5, 1.0, 2.0, 4.0), max_iter: int = 2000, seed: int = 42):
        self.c_grid = list(c_grid)
        self.max_iter = max_iter
        self.seed = seed
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.binary_model: LogisticRegression | None = None
        self.tactic_models: dict[str, LogisticRegression] = {}
        self.search_results: dict = {}

    def _fit_vectorizers(self, texts: list[str]) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
        )
        self.word_vectorizer.fit(texts)
        self.char_vectorizer.fit(texts)

    def _features(self, texts: list[str]):
        assert self.word_vectorizer is not None and self.char_vectorizer is not None, "model is not fitted"
        return hstack(
            [self.word_vectorizer.transform(texts), self.char_vectorizer.transform(texts)]
        )

    def _fit_binary(self, features, labels: np.ndarray) -> LogisticRegression:
        grid = GridSearchCV(
            LogisticRegression(class_weight="balanced", max_iter=self.max_iter, random_state=self.seed),
            {"C": self.c_grid},
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        grid.fit(features, labels)
        self.search_results["binary"] = {
            "best_C": float(grid.best_params_["C"]),
            "best_cv_f1": float(grid.best_score_),
        }
        return grid.best_estimator_

    def _fit_tactic(self, features, labels: np.ndarray, tactic: str) -> LogisticRegression | None:
        if labels.sum() < 10:  # too few positives for a stable one-vs-rest head
            return None
        grid = GridSearchCV(
            LogisticRegression(class_weight="balanced", max_iter=self.max_iter, random_state=self.seed),
            {"C": self.c_grid},
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        grid.fit(features, labels)
        self.search_results[f"tactic:{tactic}"] = {
            "best_C": float(grid.best_params_["C"]),
            "best_cv_f1": float(grid.best_score_),
        }
        return grid.best_estimator_

    def fit(self, rows: list[dict]) -> "TfidfScamModel":
        texts = [row["text"] for row in rows]
        self._fit_vectorizers(texts)
        features = self._features(texts)
        y_scam = np.array([1 if row["label"] == "scam" else 0 for row in rows])
        self.binary_model = self._fit_binary(features, y_scam)
        for tactic in TACTICS:
            y_tactic = np.array([1 if tactic in row["tactics"] else 0 for row in rows])
            model = self._fit_tactic(features, y_tactic, tactic)
            if model is not None:
                self.tactic_models[tactic] = model
        return self

    def predict_binary(self, texts: list[str]) -> np.ndarray:
        assert self.binary_model is not None, "model is not fitted"
        return self.binary_model.predict_proba(self._features(list(texts)))[:, 1]

    def _predict_heads(self, texts: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # Vectorising is ~85% of inference cost, so all seven heads share one pass.
        assert self.binary_model is not None, "model is not fitted"
        features = self._features(texts)
        tactics = {tactic: model.predict_proba(features)[:, 1] for tactic, model in self.tactic_models.items()}
        return self.binary_model.predict_proba(features)[:, 1], tactics

    def _tactic_spans(self, text: str, tactic: str) -> str:
        """The n-gram that pushed a tactic head, quoted verbatim from the message.

        Word n-grams are rebuilt from tokens, so a bigram such as "http electricity"
        can be produced from "http://electricity-bill" without appearing in the
        message. Character n-grams can likewise straddle a ``char_wb`` word padding.
        A candidate is therefore only quoted if it occurs verbatim in the message;
        otherwise the tactic is reported with generic phrasing, because the
        explanation must never cite text the sender did not write.
        """
        model = self.tactic_models[tactic]
        assert self.word_vectorizer is not None and self.char_vectorizer is not None, "model is not fitted"
        coefficients = model.coef_[0]
        word_size = len(self.word_vectorizer.vocabulary_)
        word_scores = self.word_vectorizer.transform([text]).multiply(coefficients[:word_size]).toarray().ravel()
        char_scores = self.char_vectorizer.transform([text]).multiply(coefficients[word_size:]).toarray().ravel()
        lowered = text.lower()

        word_names, char_names = self._feature_names()
        word_term = word_names[int(word_scores.argmax())]
        char_term = char_names[int(char_scores.argmax())]
        candidates = []
        if word_scores.max() > 0:
            candidates.append(word_term)
            candidates.extend(
                word for word in re.findall(r"\b\w+\b", lowered) if char_term in word and len(word) > len(char_term)
            )
        if char_scores.max() > 0:
            candidates.append(char_term)

        for candidate in candidates:
            if candidate and candidate.lower() in lowered:
                return candidate
        return "suspicious phrasing"

    def _feature_names(self) -> tuple[np.ndarray, np.ndarray]:
        # get_feature_names_out() rebuilds the whole vocabulary array on every call,
        # which made each quoted evidence span cost ~20 ms.  Built once, per process.
        cached = self.__dict__.get("_names_cache")
        if cached is None:
            assert self.word_vectorizer is not None and self.char_vectorizer is not None, "model is not fitted"
            cached = (self.word_vectorizer.get_feature_names_out(), self.char_vectorizer.get_feature_names_out())
            self.__dict__["_names_cache"] = cached
        return cached

    def __getstate__(self) -> dict:
        # Keep the cache out of the artefact: it is derived and doubles the file size.
        return {key: value for key, value in self.__dict__.items() if key != "_names_cache"}

    def predict(self, texts: list[str]) -> list[Prediction]:
        probabilities, tactic_probabilities = self._predict_heads(list(texts))
        predictions = []
        for index, text in enumerate(texts):
            tactics = []
            if probabilities[index] >= 0.5:
                for tactic, model_probabilities in tactic_probabilities.items():
                    value = float(model_probabilities[index])
                    if value >= 0.5:
                        tactics.append(
                            TacticPrediction(
                                tactic=tactic,
                                probability=value,
                            )
                        )
            predictions.append(Prediction(scam_probability=float(probabilities[index]), tactics=tactics))
        return predictions

    def evidence(self, text: str, tactic: str) -> str:
        return self._tactic_spans(text, tactic)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str) -> "TfidfScamModel":
        with open(path, "rb") as handle:
            return pickle.load(handle)
