from __future__ import annotations
import numpy as np


def as_minimization_front(income, risk):
    income = np.asarray(income, dtype=float)
    risk = np.asarray(risk, dtype=float)
    return np.column_stack([-income, risk])


def nondominated(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    keep = np.ones(len(p), dtype=bool)
    for i in range(len(p)):
        if not keep[i]:
            continue
        for j in range(len(p)):
            if i == j or not keep[i]:
                continue
            if np.all(p[j] <= p[i] + tol) and np.any(p[j] < p[i] - tol):
                keep[i] = False
    return p[keep]


def reference_scaling(reference: np.ndarray):
    ref = np.asarray(reference, dtype=float)
    ideal = np.nanmin(ref, axis=0)
    nadir = np.nanmax(ref, axis=0)
    span = np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
    return ideal, span


def normalize(points: np.ndarray, ideal: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float) - ideal) / span


def hypervolume_2d(points: np.ndarray, reference_point=(1.1, 1.1)) -> float:
    p = nondominated(np.asarray(points, dtype=float))
    rp = np.asarray(reference_point, dtype=float)
    p = p[np.all(p <= rp, axis=1)]
    if len(p) == 0:
        return 0.0
    p = p[np.argsort(p[:, 0])]
    hv = 0.0
    current_y = rp[1]
    for x, y in p:
        if y < current_y:
            hv += max(0.0, rp[0] - x) * max(0.0, current_y - y)
            current_y = y
    return float(hv)


def igd_plus(approximation: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(approximation, dtype=float)
    r = np.asarray(reference, dtype=float)
    if len(a) == 0 or len(r) == 0:
        return float('nan')
    vals = []
    for rr in r:
        d = np.sqrt(np.sum(np.maximum(a - rr, 0.0) ** 2, axis=1))
        vals.append(float(np.min(d)))
    return float(np.mean(vals))


def additive_epsilon(approximation: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(approximation, dtype=float)
    r = np.asarray(reference, dtype=float)
    if len(a) == 0 or len(r) == 0:
        return float('nan')
    return float(max(min(np.max(aa - rr) for aa in a) for rr in r))
