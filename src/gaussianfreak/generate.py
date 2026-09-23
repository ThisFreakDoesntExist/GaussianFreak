"""Sample presets from a fitted model, write them into real preset layouts and export them as a bank."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaussianfreak.fields import PERFORMANCE_DEFAULTS
from gaussianfreak.formats import (
    CATEGORIES,
    Preset,
    engine_type_value,
    write_bank,
    write_fields,
    write_preset_archive,
)
from gaussianfreak.model import CopulaPresetModel, EngineModel
from gaussianfreak.model.copula import FloatArray
from gaussianfreak.model.engine_model import to_raw_value

ENGINE_ABBREVIATIONS = {
    "BasicWaves": "Basic",
    "SuperWave": "Super",
    "Wavetable": "WTabl",
    "Harmo": "Harmo",
    "KarplusStr": "Karpl",
    "V.Analog": "VAnlg",
    "Waveshaper": "WShap",
    "Two Op. FM": "2OpFM",
    "Formant": "Frmnt",
    "Chords": "Chord",
    "Speech": "Spch",
    "Modal": "Modal",
    "Noise": "Noise",
    "Vocoder": "Vocdr",
    "Bass": "Bass",
    "SawX": "SawX",
    "Harm": "Harm",
}
NEAR_COPY_FACTOR = 0.5
"""Reject samples closer to a real preset than this share of the engine's 5th-percentile real gap."""
MIN_NEAR_COPY_DISTANCE = 0.2
"""Floor, for engines where some real presets are identical in every modelled field."""
MAX_ATTEMPTS_PER_PRESET = 20


class GenerationError(RuntimeError):
    """The model could not produce the requested presets."""


@dataclass(frozen=True, slots=True)
class GeneratedPreset:
    preset: Preset
    engine: str
    category: int
    nearest_real: str
    distance: float
    """Normalised parameter-space distance to the nearest training preset."""
    distance_percentile: float
    """Share of training presets (in %) whose own nearest-neighbour gap is at most ``distance``."""

    @property
    def category_name(self) -> str:
        return CATEGORIES[self.category]

    @property
    def filename(self) -> str:
        return f"{self.preset.name}.mfpz"


def near_copy_threshold(model: EngineModel) -> float:
    gaps = model.real_nn_distances[model.real_nn_distances > 0]
    return max(MIN_NEAR_COPY_DISTANCE, NEAR_COPY_FACTOR * float(np.percentile(gaps, 5)))


class PresetGenerator:
    """Generates presets from a fitted :class:`CopulaPresetModel`. Stateless: randomness comes from the caller."""

    def __init__(self, model: CopulaPresetModel) -> None:
        self.model = model

    def generate(
        self,
        count: int,
        rng: np.random.Generator,
        *,
        engine_id: int | None = None,
        category: int | None = None,
    ) -> list[GeneratedPreset]:
        """``count`` presets of one engine, or mixed across engines by their share of training presets.

        A category an engine has no training presets for falls back to that engine's own categories.
        """
        if engine_id is not None:
            if engine_id not in self.model.engines:
                raise GenerationError(f"no model for engine {engine_id}")
            plan: Counter[int] = Counter({engine_id: count})
        else:
            ids = list(self.model.engines)
            weights = np.array([self.model.engines[e].training_size for e in ids], dtype=float)
            plan = Counter(int(e) for e in rng.choice(ids, size=count, p=weights / weights.sum()))

        generated: list[GeneratedPreset] = []
        for eid, n in sorted(plan.items()):
            em = self.model.engines[eid]
            engine_category = category if category is not None and category in em.category_counts else None
            generated += self._generate_engine(
                em, n, rng, category=engine_category, start_index=len(generated) + 1
            )
        return generated

    def _generate_engine(
        self, em: EngineModel, count: int, rng: np.random.Generator, *, category: int | None, start_index: int
    ) -> list[GeneratedPreset]:
        threshold = near_copy_threshold(em)
        out: list[GeneratedPreset] = []
        attempts = 0
        while len(out) < count:
            attempts += 1
            if attempts > count * MAX_ATTEMPTS_PER_PRESET:
                raise GenerationError(f"{em.engine}: too many near-copies of real presets rejected")
            cat = category if category is not None else em.sample_category(rng)
            values = em.sample_values(1, rng, category=cat)
            idx, dist = em.nearest_real(values)
            if dist[0] < threshold:
                continue
            name = f"G {ENGINE_ABBREVIATIONS.get(em.engine, em.engine[:5])} {start_index + len(out):03d}"
            out.append(
                _build(em, values[0], category=cat, name=name, nearest=int(idx[0]), distance=float(dist[0]))
            )
        return out


def _build(
    em: EngineModel, values: FloatArray, *, category: int, name: str, nearest: int, distance: float
) -> GeneratedPreset:
    template = em.template
    raw = {
        key: to_raw_value(kind, float(value))
        for key, kind, value in zip(em.keys, em.kinds, values, strict=True)
        if key in template.fields
    }
    raw.update({k: v for k, v in em.typical.items() if k in template.fields})
    raw.update({k: v for k, v in PERFORMANCE_DEFAULTS.items() if k in template.fields})
    raw["VCO.Type"] = engine_type_value(em.engine_id, template.fields["VCO.Type"].metadata)
    body = write_fields(template.preset.body, raw)
    preset = template.preset.renamed(name, category).with_body(body)
    return GeneratedPreset(
        preset=preset,
        engine=em.engine,
        category=category,
        nearest_real=em.reference_names[nearest],
        distance=distance,
        distance_percentile=float(np.mean(em.real_nn_distances <= distance) * 100),
    )


@dataclass(frozen=True, slots=True)
class BankExport:
    presets_dir: Path
    bank: Path
    report: Path


def export_bank(generated: Sequence[GeneratedPreset], out_dir: Path, bank_name: str) -> BankExport:
    """Write ``<out_dir>/<bank>/*.mfpz``, ``<bank>.mfprojz`` and ``<bank>.csv``.

    Single presets are ``.mfpz``, the zipped form MIDI Control Center's preset import takes; the ``.mfprojz``
    holds the whole bank.
    """
    presets_dir = out_dir / bank_name
    presets_dir.mkdir(parents=True, exist_ok=True)
    for g in generated:
        (presets_dir / g.filename).write_bytes(write_preset_archive(g.preset))
    bank = out_dir / f"{bank_name}.mfprojz"
    bank.write_bytes(write_bank([g.preset for g in generated], bank_name))
    report = out_dir / f"{bank_name}.csv"
    with report.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["slot", "name", "engine", "category", "nearest_real_preset", "distance", "percentile"]
        )
        for slot, g in enumerate(generated, 1):
            writer.writerow(
                [
                    slot,
                    g.preset.name,
                    g.engine,
                    g.category_name,
                    g.nearest_real,
                    f"{g.distance:.3f}",
                    f"{g.distance_percentile:.0f}",
                ]
            )
    return BankExport(presets_dir=presets_dir, bank=bank, report=report)
