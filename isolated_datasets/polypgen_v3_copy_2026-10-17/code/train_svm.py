"""
SVM Training Pipeline for PolypGen with RBF Kernel, 70/15/15 Train/Val/Test Split,
and Optional Cross-Validation.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from polyp_dataset import setup_random_seed


def create_svm_pipeline(C: float = 10.0, gamma: str = "scale", random_state: Optional[int] = None) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=C, gamma=gamma, probability=True, class_weight='balanced', random_state=random_state))
    ])


def train_polyp_svm_split(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.15,
    C: float = 10.0,
    gamma: str = "scale",
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline, int]:
    """
    Trains on 70% Train split, tunes on 15% Val split, and evaluates on the 15% Test split.
    Returns (y_test_true, y_test_pred, y_test_proba, trained_pipeline, seed).
    """
    active_seed = setup_random_seed(seed)
    
    # 70% Train / 30% Holdout
    holdout_ratio = val_split + test_split
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=holdout_ratio, stratify=y, random_state=active_seed
    )
    
    # Split Holdout into 15% Val and 15% Test
    test_rel_ratio = test_split / holdout_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=test_rel_ratio, stratify=y_holdout, random_state=active_seed
    )

    print(f"[SVM-Train] Split: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%) | Seed: {active_seed}")

    pipeline = create_svm_pipeline(C=C, gamma=gamma, random_state=active_seed)
    pipeline.fit(X_train, y_train)

    val_acc = pipeline.score(X_val, y_val)
    print(f"[SVM-Train] Validation Accuracy (15% split): {val_acc*100:.2f}%")

    y_test_proba = pipeline.predict_proba(X_test)
    y_test_pred = np.argmax(y_test_proba, axis=1)

    return y_test, y_test_pred, y_test_proba, pipeline, active_seed
