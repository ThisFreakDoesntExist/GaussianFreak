"""The fitted model of one oscillator engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from gaussianfreak.dataset import PresetRecord
from gaussianfreak.fields import FieldKind, is_mod_amount
from gaussianfreak.formats import CATEGORIES, ENGINES, Preset, TaggedField, from_signed, to_signed
from gaussianfreak.model.copula import EmpiricalCopula, FloatArray
from gaussianfreak.model.destinations import DestinationSampler

_BIPOLAR_LIMIT = 32767
_UINT16_MAX = 0xFFFF

SET_SAMPLED_PREFIXES = ("EG1.", "LFO.")
"""Fields taken as a set from one training preset instead of field by field from the copula.

The envelope and the LFO are heard as one gesture, and a copula that draws each of their fields on its own
loses the shapes real presets use. Taking the whole block from one real preset of the same engine and category
keeps them, at the cost of reusing that preset's envelope and LFO."""


def to_model_value(key: str, raw: int) -> float:
    return float(to_signed(raw)) if is_mod_amount(key) else float(raw)


def to_raw_value(kind: FieldKind, value: float) -> int:
    v = round(value)
    if kind == FieldKind.BIPOLAR:
        return from_signed(max(-_BIPOLAR_LIMIT, min(_BIPOLAR_LIMIT, v)))
    return max(0, min(_UINT16_MAX, v))


def matrix_of(records: list[PresetRecord], keys: list[str]) -> FloatArray:
    """Model-space values, one row per record; NaN where a record lacks a field."""
    X = np.full((len(records), len(keys)), np.nan)
    for i, r in enumerate(records):
        for j, k in enumerate(keys):
            if k in r.fields:
                X[i, j] = to_model_value(k, r.fields[k].value)
    return X


@dataclass(frozen=True, slots=True)
class TemplatePreset:
    """The real preset generated values are written into."""

    name: str
    preset: Preset
    fields: dict[str, TaggedField]


@dataclass(slots=True)
class EngineModel:
    engine_id: int
    keys: list[str]
    kinds: list[FieldKind]
    copula: EmpiricalCopula
    template: TemplatePreset
    destinations: DestinationSampler
    reference_names: list[str]
    """Training preset names, in the row order of ``reference_values``."""
    reference_values: FloatArray
    """Training presets in model space."""
    reference_categories: list[int | None]
    """Each training preset's category, in the row order of ``reference_values``; None when it has none."""
    category_copulas: dict[int, EmpiricalCopula] = field(default_factory=dict)
    category_counts: Counter[int] = field(default_factory=Counter)
    levels: dict[int, FloatArray] = field(default_factory=dict)
    """Categorical column -> observed codes."""
    typical: dict[str, int] = field(default_factory=dict)
    """Unsampled field that real presets vary -> the engine's most common raw value."""
    scale: FloatArray = field(default_factory=lambda: np.ones(0))
    """Per-field normalisation for distances."""
    fill: FloatArray = field(default_factory=lambda: np.ones(0))
    """Per-field most common value; stands in for fields older firmware did not save."""
    real_nn_distances: FloatArray = field(default_factory=lambda: np.ones(0))
    """Each training preset's distance to its nearest other training preset."""
    _reference_normalised: FloatArray | None = None
    _donor_values: FloatArray | None = None
    _donor_rows: dict[int | None, np.ndarray] | None = None

    @property
    def engine(self) -> str:
        return ENGINES[self.engine_id] or f"engine {self.engine_id}"

    @property
    def training_size(self) -> int:
        return len(self.reference_names)

    @property
    def category_names(self) -> list[str]:
        return [CATEGORIES[c] for c, _ in self.category_counts.most_common()]

    @property
    def copula_columns(self) -> list[int]:
        return [j for j, kind in enumerate(self.kinds) if kind != FieldKind.CATEGORICAL]

    @property
    def set_sampled_columns(self) -> list[int]:
        return [j for j, key in enumerate(self.keys) if key.startswith(SET_SAMPLED_PREFIXES)]

    def donor_rows(self, category: int | None) -> np.ndarray:
        """Training presets a set-sampled block may be taken from: the category's, or all of the engine's."""
        if self._donor_rows is None:
            rows: dict[int | None, list[int]] = {None: list(range(len(self.reference_values)))}
            for row, c in enumerate(self.reference_categories):
                if c is not None:
                    rows.setdefault(c, []).append(row)
            self._donor_rows = {c: np.array(r) for c, r in rows.items()}
        return self._donor_rows.get(category, self._donor_rows[None])

    def sample_values(self, n: int, rng: np.random.Generator, category: int | None = None) -> FloatArray:
        copula = self.category_copulas.get(category, self.copula) if category is not None else self.copula
        out = np.full((n, len(self.keys)), np.nan)
        out[:, self.copula_columns] = copula.sample(n, rng)
        self.destinations.fill(out, rng)
        columns = self.set_sampled_columns
        if columns:
            if self._donor_values is None:
                self._donor_values = np.where(
                    np.isnan(self.reference_values), self.fill, self.reference_values
                )
            donors = rng.choice(self.donor_rows(category), n)
            out[:, columns] = self._donor_values[np.ix_(donors, columns)]
        return out

    def sample_category(self, rng: np.random.Generator) -> int:
        """A category drawn by its share of this engine's training presets."""
        cats, counts = zip(*self.category_counts.items(), strict=True)
        return int(rng.choice(cats, p=np.array(counts) / sum(counts)))

    def normalised(self, X: FloatArray) -> FloatArray:
        """Distance space: ordered fields scaled to their range; each categorical field one-hot, scaled so two
        different codes are exactly 1 apart whatever their numbers."""
        X = np.where(np.isnan(X), self.fill, X)
        cols = self.copula_columns
        parts = [X[:, cols] / self.scale[cols]]
        for j, codes in self.levels.items():
            parts.append((X[:, [j]] == codes[None, :]) / np.sqrt(2.0))
        return np.hstack(parts)

    def nearest_real(self, X: FloatArray) -> tuple[np.ndarray, FloatArray]:
        """Index of the nearest training preset, and its distance, for each row of model-space values."""
        if self._reference_normalised is None:
            self._reference_normalised = self.normalised(self.reference_values)
        A, B = self.normalised(X), self._reference_normalised
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
        idx = d.argmin(axis=1)
        return idx, d[np.arange(len(A)), idx]
