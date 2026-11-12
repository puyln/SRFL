from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class USGRResult:


    loss: float
    transport_cost: float
    row_penalty: float
    column_penalty: float
    coupling: Array
    random_samples: Array
    cost_matrix: Array
    iterations: int


@dataclass(frozen=True)
class GUAResult:


    parameters: Array
    weights: Array
    gram: Array
    deviation_gradients: Array
    direction: Array
    objective: float


def _as_float64_2d(name: str, value: Array) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def l2_normalize(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:


    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


def sample_spherical_gaussian(
    sample_count: int,
    feature_dim: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Array:


    if sample_count <= 0 or feature_dim <= 0:
        raise ValueError("sample_count and feature_dim must be positive")
    generator = rng if rng is not None else np.random.default_rng()
    samples = generator.normal(size=(sample_count, feature_dim))
    return l2_normalize(samples)


def project_simplex(vector: Array, mass: float = 1.0) -> Array:


    v = np.asarray(vector, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("vector must be non-empty and finite")
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be finite and positive")
    order = np.sort(v)[::-1]
    cumulative = np.cumsum(order) - mass
    indices = np.arange(1, v.size + 1)
    valid = order - cumulative / indices > 0
    if not np.any(valid):
        return np.full_like(v, mass / v.size)
    rho = indices[valid][-1]
    theta = cumulative[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def pairwise_squared_distance(x: Array, y: Array) -> Array:


    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)


def solve_unbalanced_transport(
    features: Array,
    *,
    random_sample_count: int = 128,
    tau_alpha: float = 10.0,
    tau_beta: float = 10.0,
    iterations: int = 100,
    learning_rate: Optional[float] = None,
    normalize_features: bool = True,
    rng: Optional[np.random.Generator] = None,
    random_samples: Optional[Array] = None,
) -> USGRResult:


    z = _as_float64_2d("features", features)
    if z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError("features must be non-empty")
    if not np.isfinite([tau_alpha, tau_beta]).all() or tau_alpha <= 0 or tau_beta <= 0:
        raise ValueError("tau_alpha and tau_beta must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if normalize_features:
        z = l2_normalize(z)

    if random_samples is None:
        r = sample_spherical_gaussian(
            random_sample_count, z.shape[1], rng=rng
        )
    else:
        r = _as_float64_2d("random_samples", random_samples)
        if r.shape[0] == 0:
            raise ValueError("random_samples must be non-empty")
        if r.shape[1] != z.shape[1]:
            raise ValueError(
                "features and random_samples must have the same feature dimension"
            )
        r = l2_normalize(r)

    n, m = z.shape[0], r.shape[0]
    alpha = np.full(n, 1.0 / n, dtype=np.float64)
    beta = np.full(m, 1.0 / m, dtype=np.float64)
    gamma = np.outer(alpha, beta)
    cost = pairwise_squared_distance(z, r)


    if learning_rate is None:
        learning_rate = 0.9 / (tau_alpha * m + tau_beta * n)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    for _ in range(iterations):
        row_residual = gamma.sum(axis=1) - alpha
        col_residual = gamma.sum(axis=0) - beta
        gradient = (
            cost
            + tau_alpha * row_residual[:, None]
            + tau_beta * col_residual[None, :]
        )
        gamma = np.maximum(gamma - learning_rate * gradient, 0.0)

    row_residual = gamma.sum(axis=1) - alpha
    col_residual = gamma.sum(axis=0) - beta
    transport_cost = float(np.sum(cost * gamma))
    row_penalty = float(0.5 * tau_alpha * np.dot(row_residual, row_residual))
    column_penalty = float(0.5 * tau_beta * np.dot(col_residual, col_residual))
    loss = transport_cost + row_penalty + column_penalty
    return USGRResult(
        loss=loss,
        transport_cost=transport_cost,
        row_penalty=row_penalty,
        column_penalty=column_penalty,
        coupling=gamma,
        random_samples=r,
        cost_matrix=cost,
        iterations=iterations,
    )


def _deviation_gradients(
    global_parameters: Array, client_parameters: Array, eps: float
) -> Array:
    global_vector = np.asarray(global_parameters, dtype=np.float64)
    clients = _as_float64_2d("client_parameters", client_parameters)
    if clients.size == 0 or global_vector.size == 0:
        raise ValueError("global and client parameters must be non-empty")
    if not np.isfinite(global_vector).all():
        raise ValueError("global_parameters must be finite")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if global_vector.ndim != 1:
        raise ValueError("global_parameters must be one-dimensional")
    if clients.shape[1] != global_vector.size:
        raise ValueError("global and client parameter dimensions do not match")
    difference = global_vector[None, :] - clients
    squared_deviation = np.sum(difference * difference, axis=1)
    return 2.0 * difference / np.maximum(squared_deviation[:, None], eps)


def gua_weights(
    global_parameters: Array,
    client_parameters: Array,
    *,
    iterations: int = 250,
    tolerance: float = 1e-10,
    eps: float = 1e-12,
) -> tuple[Array, Array, Array, float]:


    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    gradients = _deviation_gradients(global_parameters, client_parameters, eps)
    gram = gradients @ gradients.T
    gram = 0.5 * (gram + gram.T)
    k = gram.shape[0]
    weights = np.full(k, 1.0 / k, dtype=np.float64)
    largest_eigenvalue = float(np.linalg.eigvalsh(gram).max(initial=0.0))
    step = 1.0 / max(2.0 * largest_eigenvalue, eps)

    for _ in range(iterations):
        updated = project_simplex(weights - step * (2.0 * gram @ weights))
        if np.linalg.norm(updated - weights) <= tolerance:
            weights = updated
            break
        weights = updated

    objective = float(weights @ gram @ weights)
    return weights, gram, gradients, objective


def gua_aggregate(
    global_parameters: Array,
    client_parameters: Array,
    *,
    global_step: float = 0.1,
    solver_iterations: int = 250,
    eps: float = 1e-12,
) -> GUAResult:


    if not np.isfinite(global_step) or global_step <= 0:
        raise ValueError("global_step must be positive")
    global_vector = np.asarray(global_parameters, dtype=np.float64)
    weights, gram, gradients, objective = gua_weights(
        global_vector,
        client_parameters,
        iterations=solver_iterations,
        eps=eps,
    )
    direction = weights @ gradients
    updated = global_vector - global_step * direction
    return GUAResult(
        parameters=updated,
        weights=weights,
        gram=gram,
        deviation_gradients=gradients,
        direction=direction,
        objective=objective,
    )
