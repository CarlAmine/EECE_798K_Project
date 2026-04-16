from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq


class SINDyModel:
    def __init__(self, threshold: float = 0.01, max_iter: int = 10):
        self.threshold = threshold
        self.max_iter = max_iter
        self.coefficients: np.ndarray | None = None
        self.library_descriptions: list[str] | None = None
        self.residuals: np.ndarray | None = None

    def fit(self, library: np.ndarray, targets: np.ndarray, library_descriptions: list[str]) -> dict:
        n_terms = library.shape[1]
        n_vars = targets.shape[1]
        coefficients = np.zeros((n_terms, n_vars))
        residuals = np.zeros(n_vars)

        for var_idx in range(n_vars):
            active = np.ones(n_terms, dtype=bool)
            for _ in range(self.max_iter):
                coeff_active, _, _, _ = lstsq(library[:, active], targets[:, var_idx])
                coeff = np.zeros(n_terms)
                coeff[active] = coeff_active
                small = np.abs(coeff) < self.threshold
                small[0] = False
                updated_active = ~small
                if np.array_equal(updated_active, active):
                    active = updated_active
                    break
                active = updated_active

            coeff_active, _, _, _ = lstsq(library[:, active], targets[:, var_idx])
            coeff = np.zeros(n_terms)
            coeff[active] = coeff_active
            coefficients[:, var_idx] = coeff
            prediction = library @ coeff
            residuals[var_idx] = np.linalg.norm(targets[:, var_idx] - prediction) / (np.linalg.norm(targets[:, var_idx]) + 1e-16)

        self.coefficients = coefficients
        self.library_descriptions = library_descriptions
        self.residuals = residuals
        return {
            "residuals": residuals,
            "sparse_terms_per_var": (np.abs(coefficients) > 0).sum(axis=0),
            "total_sparse_terms": int((np.abs(coefficients) > 0).sum()),
        }

    def predict(self, library: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted yet.")
        return library @ self.coefficients

    def equations(self, var_names: list[str]) -> list[str]:
        if self.coefficients is None or self.library_descriptions is None:
            raise RuntimeError("Model has not been fitted yet.")
        equations = []
        for var_idx, var_name in enumerate(var_names):
            nonzero_idx = np.where(np.abs(self.coefficients[:, var_idx]) > 0)[0]
            if len(nonzero_idx) == 0:
                equations.append(f"d{var_name}/dt = 0")
                continue
            pieces = []
            for term_idx in nonzero_idx:
                coeff = self.coefficients[term_idx, var_idx]
                sign = "+" if coeff >= 0 else "-"
                magnitude = abs(coeff)
                chunk = f"{sign} {magnitude:.3e}*{self.library_descriptions[term_idx]}"
                pieces.append(chunk if pieces else chunk.lstrip("+ ").strip())
            equations.append(f"d{var_name}/dt = " + " ".join(pieces))
        return equations

