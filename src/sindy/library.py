from __future__ import annotations

import numpy as np


def build_polynomial_library(X: np.ndarray, degree: int = 2, var_names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    X = np.asarray(X, dtype=float)
    n_samples, n_vars = X.shape
    names = var_names or [f"x{i}" for i in range(n_vars)]

    columns = [np.ones(n_samples)]
    descriptions = ["1"]

    for index, name in enumerate(names):
        columns.append(X[:, index])
        descriptions.append(name)

    if degree >= 2:
        for first in range(n_vars):
            for second in range(first, n_vars):
                columns.append(X[:, first] * X[:, second])
                if first == second:
                    descriptions.append(f"{names[first]}^2")
                else:
                    descriptions.append(f"{names[first]}*{names[second]}")

    if degree >= 3:
        for first in range(n_vars):
            for second in range(first, n_vars):
                for third in range(second, n_vars):
                    columns.append(X[:, first] * X[:, second] * X[:, third])
                    if first == second == third:
                        descriptions.append(f"{names[first]}^3")
                    elif first == second:
                        descriptions.append(f"{names[first]}^2*{names[third]}")
                    elif second == third:
                        descriptions.append(f"{names[first]}*{names[second]}^2")
                    else:
                        descriptions.append(f"{names[first]}*{names[second]}*{names[third]}")

    return np.column_stack(columns), descriptions

