"""Fletcher-Reeves nonlinear conjugate gradient optimization."""

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
    """Compute a step size that satisfies Armijo decrease condition."""

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
    """Run Fletcher-Reeves CG and return optimization trace and convergence metrics."""

    x = np.array(x0, dtype=float)
    g = grad(x)
    direction = -g

    path: List[Array] = [x.copy()]
    values: List[float] = [f(x)]

    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break

        if np.dot(direction, g) >= 0:
            direction = -g

        alpha = backtracking_line_search(f, grad, x, direction)
        x_new = x + alpha * direction
        g_new = grad(x_new)

        beta_num = np.dot(g_new, g_new)
        beta_den = max(np.dot(g, g), 1e-20)
        beta = beta_num / beta_den

        if (k + 1) % len(x0) == 0:
            beta = 0.0

        direction = -g_new + beta * direction
        x, g = x_new, g_new

        path.append(x.copy())
        values.append(f(x))

    return {
        "path": np.array(path),
        "values": np.array(values),
        "iterations": len(values) - 1,
        "final_x": x,
        "final_value": float(values[-1]),
    }
