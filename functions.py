"""Objective functions and derivatives for unconstrained optimization demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TestFunction:
    """Container for objective function metadata and derivative callbacks."""

    name: str
    f: Callable[[Array], float]
    grad: Callable[[Array], Array]
    hess: Callable[[Array], Array]
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def sphere(x: Array) -> float:
    """Compute the Sphere function value at a 2D point."""

    return float(x[0] ** 2 + x[1] ** 2)


def sphere_grad(x: Array) -> Array:
    """Compute the gradient of the Sphere function at a 2D point."""

    return np.array([2.0 * x[0], 2.0 * x[1]], dtype=float)


def sphere_hess(_: Array) -> Array:
    """Compute the Hessian matrix of the Sphere function."""

    return np.array([[2.0, 0.0], [0.0, 2.0]], dtype=float)


def rosenbrock(x: Array) -> float:
    """Compute the Rosenbrock function value at a 2D point."""

    x1, x2 = x
    return float(100.0 * (x2 - x1**2) ** 2 + (1.0 - x1) ** 2)


def rosenbrock_grad(x: Array) -> Array:
    """Compute the gradient of the Rosenbrock function at a 2D point."""

    x1, x2 = x
    df_dx1 = -400.0 * x1 * (x2 - x1**2) - 2.0 * (1.0 - x1)
    df_dx2 = 200.0 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2], dtype=float)


def rosenbrock_hess(x: Array) -> Array:
    """Compute the Hessian matrix of the Rosenbrock function at a 2D point."""

    x1, x2 = x
    h11 = 1200.0 * x1**2 - 400.0 * x2 + 2.0
    h12 = -400.0 * x1
    h22 = 200.0
    return np.array([[h11, h12], [h12, h22]], dtype=float)


def himmelblau(x: Array) -> float:
    """Compute the Himmelblau function value at a 2D point."""

    x1, x2 = x
    return float((x1**2 + x2 - 11.0) ** 2 + (x1 + x2**2 - 7.0) ** 2)


def himmelblau_grad(x: Array) -> Array:
    """Compute the gradient of the Himmelblau function at a 2D point."""

    x1, x2 = x
    a = x1**2 + x2 - 11.0
    b = x1 + x2**2 - 7.0
    df_dx1 = 4.0 * x1 * a + 2.0 * b
    df_dx2 = 2.0 * a + 4.0 * x2 * b
    return np.array([df_dx1, df_dx2], dtype=float)


def himmelblau_hess(x: Array) -> Array:
    """Compute the Hessian matrix of the Himmelblau function at a 2D point."""

    x1, x2 = x
    h11 = 12.0 * x1**2 + 4.0 * x2 - 42.0
    h12 = 4.0 * (x1 + x2)
    h22 = 12.0 * x2**2 + 4.0 * x1 - 26.0
    return np.array([[h11, h12], [h12, h22]], dtype=float)


def get_test_functions() -> Dict[str, TestFunction]:
    """Return registry of supported test functions and plotting bounds."""

    return {
        "sphere": TestFunction(
            name="Sphere",
            f=sphere,
            grad=sphere_grad,
            hess=sphere_hess,
            xlim=(-5.0, 5.0),
            ylim=(-5.0, 5.0),
        ),
        "rosenbrock": TestFunction(
            name="Rosenbrock",
            f=rosenbrock,
            grad=rosenbrock_grad,
            hess=rosenbrock_hess,
            xlim=(-2.0, 2.0),
            ylim=(-1.0, 3.0),
        ),
        "himmelblau": TestFunction(
            name="Himmelblau",
            f=himmelblau,
            grad=himmelblau_grad,
            hess=himmelblau_hess,
            xlim=(-6.0, 6.0),
            ylim=(-6.0, 6.0),
        ),
    }
