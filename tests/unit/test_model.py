from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from gaussianfreak.dataset import PresetRecord
from gaussianfreak.fields import CATEGORICAL_FIELDS, FieldKind
from gaussianfreak.formats import CATEGORIES
from gaussianfreak.model import CopulaPresetModel, ModelConfig
from gaussianfreak.model.engine_model import matrix_of
from tests.synthetic import DESTINATIONS


def test_one_model_per_engine(model: CopulaPresetModel, records: list[PresetRecord]) -> None:
    assert set(model.engines) == {1, 3}
    assert model.training_size == len(records)


def test_fields_are_classified(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    kinds = dict(zip(em.keys, em.kinds, strict=True))
    assert kinds["VCF.Cutoff"] == FieldKind.CONTINUOUS
    assert kinds["Kbd.Octave"] == FieldKind.DISCRETE
    assert kinds["Co1.EG1"] == FieldKind.BIPOLAR
    assert kinds["Mat.Assign1"] == FieldKind.CATEGORICAL
    # performance fields, engine identity and constants are never sampled
    assert not {"Arp.Enable", "Kbd.Hold", "VCO.Type", "Pad.F00"} & set(em.keys)


def test_destinations_stay_out_of_the_copula(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    ordered = [k for k in em.keys if k not in CATEGORICAL_FIELDS]
    assert em.copula.correlation.shape == (len(ordered), len(ordered))


def test_sampled_destinations_are_real_combinations(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    cols = [em.keys.index(k) for k in ("Mat.Assign1", "Mat.Assign2", "Mat.Assign3")]
    sampled = em.sample_values(500, np.random.default_rng(0))
    assert {tuple(row) for row in sampled[:, cols]} <= set(DESTINATIONS)


def test_reference_categories_follow_the_reference_rows(
    model: CopulaPresetModel, records: list[PresetRecord]
) -> None:
    em = model.engine(3)
    assert em.reference_categories == [r.category for r in records if r.engine_id == 3]


def test_eg1_and_lfo_are_sampled_as_a_set(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    assert [em.keys[j] for j in em.set_sampled_columns] == [
        "EG1.Attack",
        "EG1.Decay",
        "LFO.Rate",
        "LFO.Shape",
    ]
    sampled = em.sample_values(200, np.random.default_rng(0))
    blocks = {tuple(row) for row in em.reference_values[:, em.set_sampled_columns]}
    assert {tuple(row) for row in sampled[:, em.set_sampled_columns]} <= blocks


def test_the_set_comes_from_a_training_preset_of_the_requested_category(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    category = CATEGORIES.index("Pad")
    rows = [i for i, c in enumerate(em.reference_categories) if c == category]
    assert rows
    sampled = em.sample_values(200, np.random.default_rng(1), category=category)
    blocks = {tuple(row) for row in em.reference_values[np.array(rows)][:, em.set_sampled_columns]}
    assert {tuple(row) for row in sampled[:, em.set_sampled_columns]} <= blocks


def test_a_category_without_training_presets_falls_back_to_the_engine(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    category = CATEGORIES.index("Strings")
    assert category not in em.category_counts
    sampled = em.sample_values(200, np.random.default_rng(2), category=category)
    blocks = {tuple(row) for row in em.reference_values[:, em.set_sampled_columns]}
    assert {tuple(row) for row in sampled[:, em.set_sampled_columns]} <= blocks


def test_other_fields_still_come_from_the_copula(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    others = [j for j in range(len(em.keys)) if j not in em.set_sampled_columns]
    sampled = em.sample_values(200, np.random.default_rng(3))
    real = {tuple(row) for row in em.reference_values[:, others]}
    assert not {tuple(row) for row in sampled[:, others]} & real


def test_correlation_is_kept(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    cutoff, reso = em.keys.index("VCF.Cutoff"), em.keys.index("VCF.Reso")
    sampled = em.sample_values(4000, np.random.default_rng(1))
    real = np.corrcoef(em.reference_values[:, cutoff], em.reference_values[:, reso])[0, 1]
    assert real < -0.5
    assert np.corrcoef(sampled[:, cutoff], sampled[:, reso])[0, 1] < -0.4


def test_category_copulas_need_enough_presets(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    assert set(em.category_copulas) == {c for c, n in em.category_counts.items() if n >= 12}
    assert em.category_copulas


def test_missing_fields_count_as_their_usual_value(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    j = em.keys.index("Kbd.Octave")
    missing = em.reference_values[:1].copy()
    usual = missing.copy()
    missing[0, j] = np.nan
    usual[0, j] = Counter(em.reference_values[:, j]).most_common(1)[0][0]
    assert np.allclose(em.normalised(missing), em.normalised(usual))


def test_distance_treats_destination_codes_as_labels(model: CopulaPresetModel) -> None:
    em = model.engine(3)
    j = em.keys.index("Mat.Assign1")
    base = em.reference_values[:1].copy()
    codes = sorted(set(em.reference_values[:, j]) - {base[0, j]})
    near, far = base.copy(), base.copy()
    near[0, j], far[0, j] = codes[0], codes[-1]

    def distance(X: np.ndarray) -> float:
        return float(np.linalg.norm(em.normalised(X) - em.normalised(base)))

    assert distance(near) > 0
    assert distance(near) == pytest.approx(distance(far))


def test_fitting_is_deterministic(records: list[PresetRecord]) -> None:
    a = CopulaPresetModel.fit(records, ModelConfig(seed=5)).engine(1)
    b = CopulaPresetModel.fit(records, ModelConfig(seed=5)).engine(1)
    assert np.array_equal(a.copula.correlation, b.copula.correlation)


def test_rejects_empty_training_set() -> None:
    with pytest.raises(ValueError, match="no training presets"):
        CopulaPresetModel.fit([])


def test_matrix_of_signs_mod_amounts(records: list[PresetRecord]) -> None:
    X = matrix_of(records, ["Co1.EG1"])
    assert np.nanmin(X) < 0


def test_save_and_load(model: CopulaPresetModel, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "model.pkl"
    model.save(path)
    loaded = CopulaPresetModel.load(path)
    assert np.array_equal(loaded.engine(3).copula.correlation, model.engine(3).copula.correlation)


def test_load_rejects_other_pickles(tmp_path: Path) -> None:
    path = tmp_path / "other.pkl"
    path.write_bytes(pickle.dumps({"not": "a model"}))
    with pytest.raises(TypeError):
        CopulaPresetModel.load(path)
