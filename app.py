"""Flask web app for visualizing unconstrained optimization algorithms."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from flask import Flask, render_template, request

import bfgs
import cg
import newton
import steepest
from functions import TestFunction, get_test_functions

app = Flask(__name__)

FUNCTIONS: Dict[str, TestFunction] = get_test_functions()
ALGORITHMS = {
    "steepest": "Steepest Descent",
    "cg": "Conjugate Gradient (Fletcher-Reeves)",
    "newton": "Newton's Method",
    "bfgs": "BFGS",
}


def parse_initial_guess(x1_raw: str, x2_raw: str) -> np.ndarray:
    """Parse and validate two scalar values into a 2D NumPy vector."""

    try:
        x1 = float(x1_raw)
        x2 = float(x2_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Initial guess values must be valid numbers.") from exc

    if not np.isfinite(x1) or not np.isfinite(x2):
        raise ValueError("Initial guess values must be finite numbers.")

    return np.array([x1, x2], dtype=float)


def run_algorithm(func: TestFunction, algorithm_key: str, x0: np.ndarray) -> Dict[str, object]:
    """Dispatch and run the selected optimization algorithm."""

    if algorithm_key == "steepest":
        return steepest.optimize(func.f, func.grad, x0)
    if algorithm_key == "cg":
        return cg.optimize(func.f, func.grad, x0)
    if algorithm_key == "newton":
        return newton.optimize(func.f, func.grad, func.hess, x0)
    if algorithm_key == "bfgs":
        return bfgs.optimize(func.f, func.grad, x0)
    raise ValueError("Unsupported algorithm selected.")


def generate_contour_plot(func: TestFunction, path: np.ndarray) -> str:
    """Create a Plotly contour plot with optimization path overlay."""

    x_min, x_max = func.xlim
    y_min, y_max = func.ylim

    x_grid = np.linspace(x_min, x_max, 140)
    y_grid = np.linspace(y_min, y_max, 140)
    xx, yy = np.meshgrid(x_grid, y_grid)

    z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            z[i, j] = func.f(np.array([xx[i, j], yy[i, j]]))

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=z,
            colorscale="Viridis",
            contours=dict(showlabels=False),
            colorbar=dict(title="f(x)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines+markers",
            line=dict(color="red", width=2),
            marker=dict(size=6),
            name="Optimization Path",
        )
    )
    fig.update_layout(
        title=f"{func.name} Contour with Optimization Path",
        xaxis_title="x1",
        yaxis_title="x2",
        template="plotly_white",
        height=500,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def generate_convergence_plot(values: np.ndarray) -> str:
    """Create a Plotly line plot of function value by iteration."""

    iterations = np.arange(len(values))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=values,
            mode="lines+markers",
            marker=dict(size=6),
            line=dict(color="#0d6efd", width=2),
            name="f(x_k)",
        )
    )
    fig.update_layout(
        title="Convergence Plot",
        xaxis_title="Iteration",
        yaxis_title="Function Value",
        template="plotly_white",
        height=420,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


@app.get("/")
def index() -> str:
    """Render the form for function, method, and initial guess selection."""

    return render_template("index.html", functions=FUNCTIONS, algorithms=ALGORITHMS)


@app.post("/optimize")
def optimize_route() -> Tuple[str, int] | str:
    """Handle optimization form submission and render visualized results."""

    function_key = request.form.get("function", "").strip()
    algorithm_key = request.form.get("algorithm", "").strip()
    x1_raw = request.form.get("x1", "").strip()
    x2_raw = request.form.get("x2", "").strip()

    try:
        if function_key not in FUNCTIONS:
            raise ValueError("Please select a valid test function.")
        if algorithm_key not in ALGORITHMS:
            raise ValueError("Please select a valid optimization algorithm.")

        x0 = parse_initial_guess(x1_raw, x2_raw)
        selected_function = FUNCTIONS[function_key]
        result = run_algorithm(selected_function, algorithm_key, x0)

        contour_div = generate_contour_plot(selected_function, result["path"])
        convergence_div = generate_convergence_plot(result["values"])

        summary = {
            "algorithm": ALGORITHMS[algorithm_key],
            "iterations": result["iterations"],
            "final_value": f"{result['final_value']:.8e}",
            "final_x": np.array2string(result["final_x"], precision=5),
        }

        return render_template(
            "result.html",
            function_name=selected_function.name,
            summary=summary,
            contour_plot=contour_div,
            convergence_plot=convergence_div,
        )
    except ValueError as exc:
        return (
            render_template(
                "index.html",
                functions=FUNCTIONS,
                algorithms=ALGORITHMS,
                error_message=str(exc),
                previous={
                    "function": function_key,
                    "algorithm": algorithm_key,
                    "x1": x1_raw,
                    "x2": x2_raw,
                },
            ),
            400,
        )
    except Exception:
        return (
            render_template(
                "index.html",
                functions=FUNCTIONS,
                algorithms=ALGORITHMS,
                error_message="Unexpected error while running optimization. Please try again.",
                previous={
                    "function": function_key,
                    "algorithm": algorithm_key,
                    "x1": x1_raw,
                    "x2": x2_raw,
                },
            ),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
