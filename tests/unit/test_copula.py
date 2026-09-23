from __future__ import annotations

import numpy as np
import pytest

from gaussianfreak.fields import FieldKind
from gaussianfreak.model.copula import EmpiricalCopula, FloatArray, nearest_correlation, spearman

KINDS = [FieldKind.CONTINUOUS, FieldKind.BIPOLAR, FieldKind.DISCRETE]


@pytest.fixture(scope="module")
def data() -> FloatArray:
    rng = np.random.default_rng(0)
    n = 600
    base = rng.normal(size=n)
    # continuous, correlated with base, with point masses at 0 and 32767
    cont = np.clip((base + rng.normal(scale=0.5, size=n)) * 12000 + 16000, 0, 32767)
    cont[rng.random(n) < 0.4] = 0
    cont[rng.random(n) < 0.15] = 32767
    # bipolar mod amount: 80% off, otherwise follows base
    mod = np.where(rng.random(n) < 0.8, 0, base * 20000).clip(-32767, 32767)
    # discrete: 4 steps driven by base
    disc = np.digitize(base, [-1, 0, 1]) * 10922
    return np.column_stack([cont, mod, disc]).astype(float)


@pytest.fixture(scope="module")
def samples(data: FloatArray) -> FloatArray:
    copula = EmpiricalCopula().fit(data, KINDS, np.random.default_rng(0))
    return copula.sample(20000, np.random.default_rng(1))


def test_point_masses_are_reproduced(data: FloatArray, samples: FloatArray) -> None:
    assert np.mean(samples[:, 0] == 0) == pytest.approx(np.mean(data[:, 0] == 0), abs=0.03)
    assert np.mean(samples[:, 0] == 32767) == pytest.approx(np.mean(data[:, 0] == 32767), abs=0.03)
    assert np.mean(samples[:, 1] == 0) == pytest.approx(np.mean(data[:, 1] == 0), abs=0.03)


def test_no_values_leak_out_of_point_masses(data: FloatArray, samples: FloatArray) -> None:
    # interpolation must not invent tiny mod amounts next to the "off" mass ...
    assert np.abs(samples[:, 1][samples[:, 1] != 0]).min() >= np.abs(data[:, 1][data[:, 1] != 0]).min()
    # ... nor values just inside a knob's 0% / 100% masses
    real = data[:, 0][(data[:, 0] > 0) & (data[:, 0] < 32767)]
    generated = samples[:, 0][(samples[:, 0] > 0) & (samples[:, 0] < 32767)]
    assert generated.min() >= real.min()
    assert generated.max() <= real.max()


def test_discrete_samples_use_only_observed_values(data: FloatArray, samples: FloatArray) -> None:
    assert set(np.unique(samples[:, 2])) <= set(np.unique(data[:, 2]))


def test_rank_correlation_is_reproduced(data: FloatArray, samples: FloatArray) -> None:
    assert np.abs(spearman(data) - spearman(samples)).max() < 0.1


def test_missing_values_are_tolerated(data: FloatArray) -> None:
    X = data.copy()
    X[:200, 0] = np.nan
    copula = EmpiricalCopula().fit(X, KINDS, np.random.default_rng(0))
    assert not np.isnan(copula.sample(1000, np.random.default_rng(3))).any()


def test_rejects_mismatched_kinds_and_empty_columns(data: FloatArray) -> None:
    with pytest.raises(ValueError, match="one kind per column"):
        EmpiricalCopula().fit(data, KINDS[:2], np.random.default_rng(0))
    X = data.copy()
    X[:, 1] = np.nan
    with pytest.raises(ValueError, match="no observations"):
        EmpiricalCopula().fit(X, KINDS, np.random.default_rng(0))


def test_nearest_correlation_is_valid() -> None:
    fixed = nearest_correlation(np.array([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]]))
    assert np.all(np.linalg.eigvalsh(fixed) > 0)
    assert np.allclose(np.diag(fixed), 1)
