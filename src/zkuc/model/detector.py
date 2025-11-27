from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from joblib import dump, load

from zkuc.model.pca_gmm import dataset_from_jsonl, fit_pca_gmm, NUMERIC_FEATURES_DEFAULT, PCAGMMModel
from zkuc.model.calibrate import fit_calibrator, Calibrator


@dataclass
class ZKUCDetector:
    """
    Compact wrapper around PCA->GMM + thin calibrator.
    """
    n_pca: int = 6
    cal_model: str = "logreg"   # "logreg" or "tree"
    n_pca_feat_for_cal: int = 3
    feature_keys: Optional[List[str]] = None

    _pga: Optional[PCAGMMModel] = None
    _cal: Optional[Calibrator] = None

    def fit_from_jsonl(self, jsonl_path: str) -> "ZKUCDetector":
        X, feat_names, y_list = dataset_from_jsonl(
            jsonl_path, feature_keys=self.feature_keys or NUMERIC_FEATURES_DEFAULT
        )
        self.feature_keys = feat_names
        # unsupervised fit
        self._pga = fit_pca_gmm(X, feat_names, n_pca=self.n_pca, random_state=0)

        # supervised calibrator on labeled subset
        mask = np.array([yy is not None for yy in y_list], dtype=bool)
        if mask.sum() >= 4:
            y = np.array([yy for yy in y_list if yy is not None], dtype=int)
            self._cal = fit_calibrator(
                self._pga, X[mask], y, n_pca_used=min(self.n_pca_feat_for_cal, self.n_pca),
                model_type=self.cal_model
            )
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._pga is None:
            raise ValueError("Detector not fit")
        scores = self._pga.score(X)
        llr = scores["llr"]
        if self._cal is not None:
            proba = self._cal.predict_proba(self._pga, X)
        else:
            proba = scores["buggy_post"]
        return proba, llr

    def score_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        X, feat_names, y_list, ids = dataset_from_jsonl(
            jsonl_path, feature_keys=self.feature_keys, return_ids=True
        )
        proba, llr = self.predict(X)
        rows = []
        for rid, pr, l, yy in zip(ids, proba, llr, y_list):
            row = {"id": rid, "prob_buggy": float(pr), "llr": float(l)}
            if yy is not None:
                row["label"] = int(yy)
            rows.append(row)
        return rows

    def save(self, path: str):
        dump(
            {
                "cfg": {
                    "n_pca": self.n_pca,
                    "cal_model": self.cal_model,
                    "n_pca_feat_for_cal": self.n_pca_feat_for_cal,
                    "feature_keys": self.feature_keys,
                },
                "pga": self._pga,
                "cal": self._cal,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "ZKUCDetector":
        blob = load(path)
        det = ZKUCDetector(**blob["cfg"])
        det._pga = blob["pga"]
        det._cal = blob["cal"]
        return det
