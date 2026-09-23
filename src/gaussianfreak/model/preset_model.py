"""Fit one copula model per oscillator engine.

For each engine the model keeps every sound field's real distribution (exact point masses at 0 / 100% and for
unused mod routes) and a Gaussian copula for how fields move together. Engines with few presets borrow
correlation structure from a copula pooled over all engines (shrinkage), except for the oscillator knobs, whose
meaning is engine-specific. Assignable mod destinations are labels, not amounts: they stay out of the copula and
are drawn as whole real combinations. Every engine writes its presets into the same Init preset.
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaussianfreak.dataset import PresetRecord
from gaussianfreak.fields import (
    CATEGORICAL_FIELDS,
    ENGINE_SPECIFIC_FIELDS,
    PITCH_MAPPING_FIELDS,
    FieldKind,
    classify_field,
    is_sound_field,
)
from gaussianfreak.model.copula import EmpiricalCopula, FloatArray, nearest_correlation
from gaussianfreak.model.destinations import DestinationSampler
from gaussianfreak.model.engine_model import EngineModel, TemplatePreset, matrix_of
from gaussianfreak.model.template import init_template

MIN_CATEGORY_OBSERVATIONS = 12
"""Presets of one category an engine needs before that category gets its own field distributions."""
MIN_PRESENCE = 0.5
MIN_PRESENT_PRESETS = 30
"""A field is modelled if at least half an engine's presets have it, or at least this many do: newer firmware's
unison/chord settings exist in only 26-47% of an engine's presets."""
_FULL_SCALE = 32767.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    seed: int = 0
    shrinkage_strength: int = 50
    """Pooled correlation weight is ``k / (k + engine presets)``."""


class CopulaPresetModel:
    """Per-engine copula models fitted on training presets."""

    def __init__(self, engines: dict[int, EngineModel], config: ModelConfig) -> None:
        self.engines = engines
        self.config = config

    @classmethod
    def fit(cls, records: Sequence[PresetRecord], config: ModelConfig | None = None) -> CopulaPresetModel:
        config = config or ModelConfig()
        return cls(_Fitter(config).fit(list(records)), config)

    def engine(self, engine_id: int) -> EngineModel:
        return self.engines[engine_id]

    @property
    def training_size(self) -> int:
        return sum(m.training_size for m in self.engines.values())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> CopulaPresetModel:
        """Load a model written by :meth:`save`. Models are pickles: only load files you trained yourself."""
        with path.open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not a trained model")
        return model


class _Fitter:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.template = init_template()

    def fit(self, records: list[PresetRecord]) -> dict[int, EngineModel]:
        if not records:
            raise ValueError("no training presets")
        by_engine: dict[int, list[PresetRecord]] = defaultdict(list)
        for r in records:
            by_engine[r.engine_id].append(r)

        pooled_keys = [k for k in _sound_keys(records) if k not in CATEGORICAL_FIELDS]
        pooled = EmpiricalCopula().fit(
            matrix_of(records, pooled_keys), _kinds(records, pooled_keys), self.rng
        )
        pooled_index = {k: i for i, k in enumerate(pooled_keys)}

        engines = {}
        for engine_id, recs in sorted(by_engine.items()):
            engines[engine_id] = self._fit_engine(engine_id, recs, records, pooled, pooled_index)
        return engines

    def _fit_engine(
        self,
        engine_id: int,
        recs: list[PresetRecord],
        all_records: list[PresetRecord],
        pooled: EmpiricalCopula,
        pooled_index: dict[str, int],
    ) -> EngineModel:
        keys = _sound_keys(recs)
        kinds = _kinds(recs, keys)
        X = matrix_of(recs, keys)
        ordered = [j for j, kind in enumerate(kinds) if kind != FieldKind.CATEGORICAL]
        ordered_keys = [keys[j] for j in ordered]
        copula = EmpiricalCopula().fit(X[:, ordered], [kinds[j] for j in ordered], self.rng)
        idx = [pooled_index[k] for k in ordered_keys]
        R_pool = pooled.correlation[np.ix_(idx, idx)].copy()
        for a, ka in enumerate(ordered_keys):
            for b, kb in enumerate(ordered_keys):
                if a != b and (ka in ENGINE_SPECIFIC_FIELDS or kb in ENGINE_SPECIFIC_FIELDS):
                    R_pool[a, b] = 0.0
        weight = self.config.shrinkage_strength / (self.config.shrinkage_strength + len(recs))
        copula.correlation = nearest_correlation((1 - weight) * copula.correlation + weight * R_pool)

        model = EngineModel(
            engine_id=engine_id,
            keys=keys,
            kinds=kinds,
            copula=copula,
            template=self.template,
            destinations=DestinationSampler(keys, recs, all_records),
            reference_names=[r.name for r in recs],
            reference_values=X,
            reference_categories=[r.category for r in recs],
        )
        model.levels = {
            j: np.unique(X[~np.isnan(X[:, j]), j])
            for j, kind in enumerate(kinds)
            if kind == FieldKind.CATEGORICAL
        }
        model.category_counts = Counter(r.category for r in recs if r.category is not None)
        for category, count in model.category_counts.items():
            if count >= MIN_CATEGORY_OBSERVATIONS:
                subset = [r for r in recs if r.category == category]
                model.category_copulas[category] = copula.with_marginals(
                    matrix_of(subset, keys)[:, ordered], MIN_CATEGORY_OBSERVATIONS
                )
        model.typical = _typical_values(recs, keys, self.template)
        model.scale = _scales(kinds, X)
        model.fill = np.array(
            [Counter(X[~np.isnan(X[:, j]), j]).most_common(1)[0][0] for j in range(len(keys))]
        )
        model.real_nn_distances = _leave_one_out_nn(model, X)
        return model


def _sound_keys(records: list[PresetRecord]) -> list[str]:
    presence = Counter(k for r in records for k in r.fields if is_sound_field(k))
    keys = []
    for k, count in sorted(presence.items()):
        if count / len(records) < MIN_PRESENCE and count < MIN_PRESENT_PRESETS:
            continue
        if (
            len({r.fields[k].value for r in records if k in r.fields}) > 1
        ):  # constants are filled in as typical values
            keys.append(k)
    return keys


def _kinds(records: list[PresetRecord], keys: list[str]) -> list[FieldKind]:
    kinds = []
    for k in keys:
        present = [r.fields[k] for r in records if k in r.fields]
        kinds.append(classify_field(k, [f.metadata for f in present], len({f.value for f in present})))
    return kinds


def _typical_values(records: list[PresetRecord], keys: list[str], template: TemplatePreset) -> dict[str, int]:
    """Sound and pitch-mapping fields the model does not sample: the engine's most common raw value, so generated
    presets never keep the Init preset's value where real presets of the engine use another (constants included)."""
    typical = {}
    for key in template.fields:
        if key in keys or not (is_sound_field(key) or key in PITCH_MAPPING_FIELDS):
            continue
        values = Counter(r.fields[key].value for r in records if key in r.fields)
        if values:
            typical[key] = values.most_common(1)[0][0]
    return typical


def _scales(kinds: list[FieldKind], X: FloatArray) -> FloatArray:
    scale = np.full(len(kinds), _FULL_SCALE)
    for j, kind in enumerate(kinds):
        if kind == FieldKind.DISCRETE:
            observed = X[~np.isnan(X[:, j]), j]
            span = observed.max() - observed.min()
            scale[j] = span if span > 0 else 1.0
    return scale


def _leave_one_out_nn(model: EngineModel, X: FloatArray) -> FloatArray:
    B = model.normalised(X)
    d = np.sqrt(((B[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    result: FloatArray = d.min(axis=1)
    return result
