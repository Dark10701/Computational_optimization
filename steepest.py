"""Steepest descent optimization with Armijo backtracking line search."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

Array = np.ndarray


def backtracking_line_search(
    f: Callable[[Array], float],
    grad: Callable[[Array], Array],
    x: Array,
    direction: Array,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
) -> float:
    """Compute a step size that satisfies the Armijo decrease condition."""

    alpha = alpha0
    fx = f(x)
    gx = grad(x)
    while f(x + alpha * direction) > fx + c * alpha * np.dot(gx, direction):
        alpha *= rho
        if alpha < 1e-12:
            break
    return alpha


def optimize(
    f: Callable[[Array], float],
    grad: Callable[[Array], Array],
    x0: Array,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> Dict[str, object]:
    """Run steepest descent and return optimization trace and convergence metrics."""

    x = np.array(x0, dtype=float)
    path: List[Array] = [x.copy()]
    values: List[float] = [f(x)]

    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        direction = -g
        alpha = backtracking_line_search(f, grad, x, direction)
        x = x + alpha * direction
        path.append(x.copy())
        values.append(f(x))

    return {
        "path": np.array(path),
        "values": np.array(values),
        "iterations": len(values) - 1,
        "final_x": x,
        "final_value": float(values[-1]),
    }
