"""Morgan fingerprint baselines with one estimator per property task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass(frozen=True)
class MorganConfig:
    estimator: str = "rf"
    radius: int = 2
    n_bits: int = 2048
    n_estimators: int = 500
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 30
    seed: int = 0
    n_jobs: int = -1


class MorganFingerprintModel:
    """Morgan + RF/XGBoost supporting regression and masked multitask labels."""

    def __init__(self, task_type: str, num_tasks: int, config: MorganConfig) -> None:
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")
        if config.estimator not in {"rf", "xgboost"}:
            raise ValueError("estimator must be 'rf' or 'xgboost'")
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.config = config
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=config.radius, fpSize=config.n_bits
        )
        self.estimators: list[object | None] = []

    def featurize(self, smiles: list[str]) -> np.ndarray:
        features = np.zeros((len(smiles), self.config.n_bits), dtype=np.uint8)
        for index, text in enumerate(smiles):
            molecule = Chem.MolFromSmiles(text)
            if molecule is None:
                raise ValueError(f"Invalid SMILES at row {index}: {text!r}")
            features[index] = self.generator.GetFingerprintAsNumPy(molecule)
        return features

    def _new_estimator(self, task: int):
        seed = self.config.seed + task
        if self.config.estimator == "rf":
            cls = RandomForestRegressor if self.task_type == "regression" else RandomForestClassifier
            options = dict(
                n_estimators=self.config.n_estimators,
                random_state=seed,
                n_jobs=self.config.n_jobs,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
            )
            if self.task_type == "classification":
                options["class_weight"] = "balanced"
            return cls(**options)

        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError("Morgan + XGBoost requires the xgboost package") from exc
        cls = XGBRegressor if self.task_type == "regression" else XGBClassifier
        options = dict(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth or 6,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=self.config.n_jobs,
            tree_method="hist",
        )
        if self.task_type == "regression":
            options["objective"] = "reg:squarederror"
        else:
            options.update(objective="binary:logistic", eval_metric="logloss")
        return cls(**options)

    def fit(
        self,
        smiles: list[str],
        targets: np.ndarray,
        target_mask: np.ndarray,
        validation_smiles: list[str] | None = None,
        validation_targets: np.ndarray | None = None,
        validation_mask: np.ndarray | None = None,
    ) -> "MorganFingerprintModel":
        features = self.featurize(smiles)
        validation_features = (
            self.featurize(validation_smiles) if validation_smiles is not None else None
        )
        self.estimators = []
        for task in range(self.num_tasks):
            observed = target_mask[:, task].astype(bool)
            if not observed.any():
                self.estimators.append(None)
                continue
            values = targets[observed, task]
            if self.task_type == "classification" and np.unique(values).size < 2:
                self.estimators.append(None)
                continue
            estimator = self._new_estimator(task)
            fit_options = {}
            if self.config.estimator == "xgboost" and validation_features is not None:
                if validation_targets is None or validation_mask is None:
                    raise ValueError("validation targets and mask must accompany validation SMILES")
                validation_observed = validation_mask[:, task].astype(bool)
                if validation_observed.any():
                    estimator.set_params(
                        early_stopping_rounds=self.config.early_stopping_rounds
                    )
                    fit_options["eval_set"] = [
                        (
                            validation_features[validation_observed],
                            validation_targets[validation_observed, task],
                        )
                    ]
                    fit_options["verbose"] = False
            if self.task_type == "classification" and self.config.estimator == "xgboost":
                positives = max(float((values > 0.5).sum()), 1.0)
                negatives = float((values <= 0.5).sum())
                estimator.set_params(scale_pos_weight=negatives / positives)
            estimator.fit(features[observed], values, **fit_options)
            self.estimators.append(estimator)
        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        if len(self.estimators) != self.num_tasks:
            raise RuntimeError("fit must be called before predict")
        features = self.featurize(smiles)
        output = np.full((len(smiles), self.num_tasks), np.nan, dtype=np.float32)
        for task, estimator in enumerate(self.estimators):
            if estimator is None:
                continue
            if self.task_type == "classification":
                probability = estimator.predict_proba(features)[:, 1]
                probability = np.clip(probability, 1e-7, 1 - 1e-7)
                output[:, task] = np.log(probability / (1 - probability))
            else:
                output[:, task] = estimator.predict(features)
        return output
