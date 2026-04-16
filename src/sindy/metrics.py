from __future__ import annotations

import numpy as np
from scipy.integrate import odeint


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-16))


def rollout_polynomial(coefficients: np.ndarray, descriptions: list[str], x0: np.ndarray, t: np.ndarray, variable_aliases: dict[str, str]) -> np.ndarray:
    def rhs(state: np.ndarray, _: float) -> np.ndarray:
        values = {
            "1": 1.0,
            variable_aliases["state_1"]: state[0],
            variable_aliases["state_2"]: state[1],
        }
        phi_terms = []
        for term in descriptions:
            if term == "1":
                phi_terms.append(1.0)
                continue
            value = 1.0
            for factor in term.split("*"):
                if "^" in factor:
                    base_name, exponent = factor.split("^")
                    value *= values[base_name] ** int(exponent)
                else:
                    value *= values[factor]
            phi_terms.append(value)
        return np.array(phi_terms) @ coefficients

    return odeint(rhs, x0, t)


def compute_rollout_metrics(x_true: np.ndarray, x_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(x_true, x_pred),
        "mae": mae(x_true, x_pred),
        "relative_error": relative_error(x_true, x_pred),
    }
