"""Gaussian copula with empirical marginals.

Each column keeps its exact observed distribution, including point masses such as "84% of mod-matrix cells are
0" or "knob at 100%"; a Gaussian copula over normal scores carries the rank correlations between columns.
Missing values (NaN) are ignored per column and per pair.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from math import erf, sqrt
from statistics import NormalDist
from typing import Self

import numpy as np
from numpy.typing import NDArray

from gaussianfreak.fields import FieldKind

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_ppf = np.vectorize(NormalDist().inv_cdf, otypes=[float])
_erf = np.vectorize(erf, otypes=[float])

MIN_PAIR_OBSERVATIONS = 10
MASS_MIN_COUNT = 3
"""A value repeated at least this often (and in at least MASS_MIN_SHARE of rows) is a point mass."""
MASS_MIN_SHARE = 0.02
CALIBRATION_TOLERANCE = 0.01
CALIBRATION_SAMPLES = 8000


def normal_cdf(z: FloatArray) -> FloatArray:
    result: FloatArray = 0.5 * (1.0 + _erf(z / sqrt(2.0)))
    return result


def average_ranks(values: FloatArray) -> FloatArray:
    """1-based ranks with ties sharing their average rank."""
    order = np.argsort(values, kind="mergesort")
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    ranks = np.empty(len(values))
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    return ranks


def spearman(X: FloatArray) -> FloatArray:
    """Spearman rank correlation matrix; columns without variation correlate 0."""
    R = np.column_stack([average_ranks(X[:, j]) for j in range(X.shape[1])])
    R = R - R.mean(axis=0)
    sd = R.std(axis=0)
    sd[sd == 0] = np.inf
    Z = R / sd
    result: FloatArray = np.einsum("ki,kj->ij", Z, Z) / len(X)
    return result


def mass_flags(sorted_values: FloatArray) -> BoolArray:
    """True where a sorted value belongs to a point mass (e.g. 'off', 0%, 100%)."""
    _, inverse, counts = np.unique(sorted_values, return_inverse=True, return_counts=True)
    needed = max(MASS_MIN_COUNT, int(np.ceil(MASS_MIN_SHARE * len(sorted_values))))
    flags: BoolArray = counts[inverse] >= needed
    return flags


def pairwise_spearman(X: FloatArray) -> tuple[FloatArray, BoolArray]:
    """Spearman matrix over pairwise-complete rows, and which pairs had enough data."""
    d = X.shape[1]
    R = np.eye(d)
    ok = np.zeros((d, d), dtype=bool)
    present = ~np.isnan(X)
    for i in range(d):
        for j in range(i + 1, d):
            rows = present[:, i] & present[:, j]
            if rows.sum() < MIN_PAIR_OBSERVATIONS:
                continue
            R[i, j] = R[j, i] = spearman(X[rows][:, [i, j]])[0, 1]
            ok[i, j] = ok[j, i] = True
    return R, ok


def nearest_correlation(R: FloatArray, floor: float = 1e-6) -> FloatArray:
    """Clip negative eigenvalues and rescale to a unit diagonal."""
    R = (R + R.T) / 2
    w, V = np.linalg.eigh(R)
    fixed = np.einsum("ik,k,jk->ij", V, np.maximum(w, floor), V)
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    result: FloatArray = fixed
    return result


class EmpiricalCopula:
    """Empirical marginals joined by a Gaussian copula."""

    def __init__(self) -> None:
        self.kinds: list[FieldKind] = []
        self.sorted_values: list[FloatArray] = []
        self.masses: list[BoolArray] = []
        self.correlation: FloatArray = np.eye(0)

    @property
    def dimension(self) -> int:
        return len(self.kinds)

    def fit(
        self,
        X: FloatArray,
        kinds: Sequence[FieldKind],
        rng: np.random.Generator,
        *,
        calibration_rounds: int = 8,
    ) -> Self:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if len(kinds) != d:
            raise ValueError("one kind per column required")
        self.kinds = list(kinds)
        self.sorted_values = []
        self.masses = []
        Z = np.full((n, d), np.nan)
        for j in range(d):
            mask = ~np.isnan(X[:, j])
            observed = X[mask, j]
            if len(observed) == 0:
                raise ValueError(f"column {j} has no observations")
            self.sorted_values.append(np.sort(observed))
            self.masses.append(mass_flags(self.sorted_values[-1]))
            Z[mask, j] = _ppf((average_ranks(observed) - 0.5) / len(observed))
        self.correlation = nearest_correlation(_pairwise_pearson(Z))
        if calibration_rounds:
            self._calibrate(X, calibration_rounds, rng)
        return self

    def _calibrate(self, X: FloatArray, rounds: int, rng: np.random.Generator) -> None:
        """Point masses weaken rank correlations after sampling; nudge the copula correlation until simulated
        Spearman correlations match the observed (pairwise-complete) ones."""
        target, observed = pairwise_spearman(X)
        for _ in range(rounds):
            simulated = spearman(self.sample(CALIBRATION_SAMPLES, rng))
            gap = np.where(observed, target - simulated, 0.0)
            np.fill_diagonal(gap, 0.0)
            if np.abs(gap).max() < CALIBRATION_TOLERANCE:
                break
            self.correlation = nearest_correlation(np.clip(self.correlation + gap, -0.99, 0.99))

    def with_marginals(self, X_subset: FloatArray, min_observations: int) -> EmpiricalCopula:
        """Copy keeping this correlation, with marginals from a subset wherever it has enough data."""
        other = deepcopy(self)
        for j in range(X_subset.shape[1]):
            observed = X_subset[~np.isnan(X_subset[:, j]), j]
            if len(observed) >= min_observations:
                other.sorted_values[j] = np.sort(observed)
                other.masses[j] = mass_flags(other.sorted_values[j])
        return other

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        L = np.linalg.cholesky(self.correlation)
        # einsum instead of @: avoids spurious BLAS floating-point warnings on macOS Accelerate
        U = normal_cdf(np.einsum("ij,kj->ik", rng.standard_normal((n, self.dimension)), L))
        out = np.empty((n, self.dimension))
        for j in range(self.dimension):
            out[:, j] = self._quantile(j, U[:, j])
        return out

    def _quantile(self, j: int, u: FloatArray) -> FloatArray:
        s = self.sorted_values[j]
        m = len(s)
        if self.kinds[j] == FieldKind.DISCRETE:
            discrete: FloatArray = s[np.minimum((u * m).astype(int), m - 1)]
            return discrete
        pos = np.clip(u * m - 0.5, 0, m - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, m - 1)
        frac = pos - lo
        value = s[lo] + frac * (s[hi] - s[lo])
        # never interpolate out of a point mass: that would invent tiny mod amounts or near-0% knobs
        at_mass_edge = (s[lo] != s[hi]) & (self.masses[j][lo] | self.masses[j][hi])
        result: FloatArray = np.where(at_mass_edge, np.where(frac < 0.5, s[lo], s[hi]), value)
        return result


def _pairwise_pearson(Z: FloatArray) -> FloatArray:
    d = Z.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            rows = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            if rows.sum() < MIN_PAIR_OBSERVATIONS:
                continue
            a, b = Z[rows, i], Z[rows, j]
            sa, sb = a.std(), b.std()
            if sa == 0 or sb == 0:
                continue
            R[i, j] = R[j, i] = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    return R
