"""End-to-end checks on whatever presets are under training/. Skipped when it holds no preset files.

The statistical checks need enough presets of one engine to mean anything; engines or fields a library is too
small for are skipped rather than failed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from gaussianfreak.dataset import PresetLibrary, PresetRecord
from gaussianfreak.fields import ASSIGN_COLUMNS, PITCH_MAPPING_FIELDS, is_sound_field
from gaussianfreak.formats import ENGINES, Preset, read_fields
from gaussianfreak.generate import PresetGenerator
from gaussianfreak.model import CopulaPresetModel, ModelConfig
from gaussianfreak.model.engine_model import EngineModel, matrix_of
from gaussianfreak.train import train

TRAINING_DIR = Path(__file__).resolve().parents[2] / "training"
SUPERWAVE, WAVETABLE, HARMO, V_ANALOG, TWO_OP_FM = 2, 3, 4, 6, 8
MIN_ENGINE_PRESETS = 30
"""Presets an engine needs before its sampled statistics are compared with the real ones."""

pytestmark = [
    pytest.mark.training_data,
    pytest.mark.skipif(
        not any(p.is_file() and p.name != "README.md" for p in TRAINING_DIR.rglob("*")),
        reason="no training data under training/",
    ),
]


@pytest.fixture(scope="module")
def real_records() -> list[PresetRecord]:
    records = PresetLibrary(TRAINING_DIR).scan().records
    if not records:
        pytest.skip("no training-ready presets under training/")
    return records


@pytest.fixture(scope="module")
def real_model(real_records: list[PresetRecord]) -> CopulaPresetModel:
    return CopulaPresetModel.fit(real_records, ModelConfig(seed=11))


def engine_model(
    model: CopulaPresetModel, engine_id: int, min_presets: int = MIN_ENGINE_PRESETS
) -> EngineModel:
    """The engine's model, or skip when the library has too few of its presets."""
    em = model.engines.get(engine_id)
    if em is None or em.training_size < min_presets:
        pytest.skip(f"{ENGINES[engine_id]}: needs {min_presets}+ presets under training/")
    return em


def field_index(em: EngineModel, key: str) -> int:
    """Column of a modelled field, or skip when the library leaves it unmodelled (e.g. it never varies)."""
    if key not in em.keys:
        pytest.skip(f"{key} is not modelled on {em.engine} with this library")
    return em.keys.index(key)


def test_every_training_preset_round_trips_byte_for_byte(real_records: list[PresetRecord]) -> None:
    assert all(Preset.from_bytes(r.preset.to_bytes()) == r.preset for r in real_records)


def test_one_model_per_engine_in_the_library(
    real_model: CopulaPresetModel, real_records: list[PresetRecord]
) -> None:
    assert set(real_model.engines) == {r.engine_id for r in real_records}


def test_generated_values_stay_within_real_ranges(real_model: CopulaPresetModel) -> None:
    em = engine_model(real_model, TWO_OP_FM, min_presets=1)
    generated = PresetGenerator(real_model).generate(40, np.random.default_rng(5), engine_id=TWO_OP_FM)
    values = np.array(
        [
            [
                read_fields(g.preset.body)[k].value if k in read_fields(g.preset.body) else np.nan
                for k in em.keys
            ]
            for g in generated
        ],
        dtype=float,
    )
    signed = [j for j, k in enumerate(em.keys) if k.startswith("Co")]
    values[:, signed] = np.where(values[:, signed] >= 0x8000, values[:, signed] - 0x10000, values[:, signed])
    lo, hi = np.nanmin(em.reference_values, axis=0), np.nanmax(em.reference_values, axis=0)
    assert np.all((values >= lo - 1) & (values <= hi + 1))


def test_unsampled_fields_take_the_engines_most_common_value(
    real_model: CopulaPresetModel, real_records: list[PresetRecord]
) -> None:
    # the Init preset's values must never leak into fields real presets of the engine set differently
    checked = 0
    for engine_id in (HARMO, V_ANALOG, TWO_OP_FM):
        em = real_model.engines.get(engine_id)
        if em is None:
            continue
        engine_records = [r for r in real_records if r.engine_id == engine_id]
        generated = PresetGenerator(real_model).generate(10, np.random.default_rng(2), engine_id=engine_id)
        for key in em.template.fields:
            if key in em.keys or not (is_sound_field(key) or key in PITCH_MAPPING_FIELDS):
                continue
            values = Counter(r.fields[key].value for r in engine_records if key in r.fields)
            if len(values) < 2:
                continue
            checked += 1
            typical = values.most_common(1)[0][0]
            assert all(read_fields(g.preset.body)[key].value == typical for g in generated), (em.engine, key)
    if not checked:
        pytest.skip("no unsampled field varies across this library's presets")


def test_octave_is_sampled_like_real_presets(real_model: CopulaPresetModel) -> None:
    em = engine_model(real_model, WAVETABLE)
    j = field_index(em, "Kbd.Octave")
    real = em.reference_values[:, j]
    sampled = em.sample_values(3000, np.random.default_rng(4))[:, j]
    assert np.mean(sampled == 16384) == pytest.approx(np.mean(real == 16384), abs=0.08)
    assert np.mean(sampled < 16384) == pytest.approx(np.mean(real < 16384), abs=0.08)


def test_newer_firmware_unison_and_chord_settings_keep_their_variety(real_model: CopulaPresetModel) -> None:
    em = engine_model(real_model, SUPERWAVE)
    for key in ("Gen.Chord", "Gen.UniSprd"):
        j = field_index(em, key)
        real = em.reference_values[:, j]
        real = real[~np.isnan(real)]
        typical = Counter(real).most_common(1)[0][0]
        sampled = em.sample_values(3000, np.random.default_rng(6))[:, j]
        assert np.mean(sampled != typical) == pytest.approx(np.mean(real != typical), abs=0.06), key


def test_mod_matrix_density_matches_real(real_model: CopulaPresetModel) -> None:
    em = engine_model(real_model, TWO_OP_FM)
    mod = [j for j, k in enumerate(em.keys) if k.startswith("Co")]
    real_active = np.mean(np.sum(np.abs(np.nan_to_num(em.reference_values[:, mod])) > 0, axis=1))
    sampled = em.sample_values(3000, np.random.default_rng(9))[:, mod]
    assert np.mean(np.sum(np.abs(sampled) > 0.5, axis=1)) == pytest.approx(real_active, rel=0.2)


def test_destinations_are_real_combinations_that_follow_their_mod_column(
    real_model: CopulaPresetModel, real_records: list[PresetRecord]
) -> None:
    em = engine_model(real_model, TWO_OP_FM)
    slots = [k for k in sorted(ASSIGN_COLUMNS) if k in em.keys]
    if not slots:
        pytest.skip(f"no assignable mod destination varies on {em.engine} with this library")
    sampled = em.sample_values(3000, np.random.default_rng(3))
    real_trios = {
        tuple(r.fields[k].value for k in slots) for r in real_records if all(k in r.fields for k in slots)
    }
    cols = [em.keys.index(k) for k in slots]
    assert {tuple(int(v) for v in row) for row in sampled[:, cols]} <= real_trios

    real = matrix_of([r for r in real_records if r.engine_id == TWO_OP_FM], em.keys)
    for key in slots:
        j = em.keys.index(key)
        mods = [i for i, k in enumerate(em.keys) if k.startswith(ASSIGN_COLUMNS[key] + ".")]
        default = Counter(real[:, j]).most_common(1)[0][0]
        shares = []
        for X in (real, sampled):
            active = (np.abs(np.nan_to_num(X[:, mods])) > 0.5).any(axis=1)
            shares.append((np.mean(X[active, j] == default), np.mean(X[~active, j] == default)))
        assert shares[1] == pytest.approx(shares[0], abs=0.1), key


def test_train_save_and_generate(tmp_path: Path, real_records: list[PresetRecord]) -> None:
    model, _ = train(TRAINING_DIR, tmp_path / "model.pkl", seed=0)
    assert model.training_size == len(real_records)
    loaded = CopulaPresetModel.load(tmp_path / "model.pkl")
    # the library's most common engine, in that engine's most common category
    engine_id = Counter(r.engine_id for r in real_records).most_common(1)[0][0]
    em = loaded.engine(engine_id)
    if not em.category_counts:
        pytest.skip(f"no {em.engine} preset under training/ has a category")
    category = em.category_counts.most_common(1)[0][0]
    [preset] = PresetGenerator(loaded).generate(
        1, np.random.default_rng(1), engine_id=engine_id, category=category
    )
    assert preset.engine == em.engine
    assert preset.category == category
