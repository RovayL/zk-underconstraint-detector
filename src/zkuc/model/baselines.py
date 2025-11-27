# src/zkuc/model/baselines.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

@dataclass
class Featurizer:
    use_pca: bool = True
    n_pca: int = 6

    def fit(self, X: np.ndarray):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        if self.use_pca:
            self.pca = PCA(self.n_pca).fit(Z)
        else:
            self.pca = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        return self.pca.transform(Z) if self.pca is not None else Z

def gaussian_nb():
    return GaussianNB()

def logreg():
    return LogisticRegression(max_iter=200, class_weight="balanced")

def linear_svm():
    # decision_function supported; probability=False is fine for ROC/PR
    return LinearSVC(class_weight="balanced")

def rbf_svm():
    # probability=True enables calibrated probabilities (Platt)
    return SVC(kernel="rbf", probability=True, class_weight="balanced")
