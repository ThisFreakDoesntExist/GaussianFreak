"""Synthetic presets with real body layouts, so tests run without any training data."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from gaussianfreak.formats import UNPACKED_LEN, Preset, pack_7bit, write_bank

ENGINE_COUNT = 22
FILLER_FIELDS = 90  # real layouts have 96-110 tagged fields
CONTINUOUS_META = 0
OCTAVE_META = 6
OCTAVES = (5461, 10922, 16384, 21845)
DESTINATIONS = ((2306.0, 2307.0, 2308.0), (100.0, 200.0, 300.0), (512.0, 513.0, 514.0))


def build_body(fields: dict[str, tuple[int, int]]) -> bytes:
    """A packed body holding ``{"GRP.Name": (metadata, uint16 value)}`` in the device's tagged layout."""
    out = bytearray()
    group = None
    for key, (meta, value) in fields.items():
        grp, name = key.split(".", 1)
        if grp != group:
            out += (b"#" if group is None else b"@#") + grp.encode()
            group = grp
        out += bytes([0x40 + len(name)]) + name.encode() + b"c" + bytes([meta, value & 0xFF, value >> 8])
    if len(out) > UNPACKED_LEN:
        raise ValueError("too many fields")
    return pack_7bit(bytes(out) + bytes(UNPACKED_LEN - len(out)))


def engine_value(engine_id: int) -> int:
    return round(engine_id * 32767 / ENGINE_COUNT)


def signed(value: int) -> int:
    return value + 0x10000 if value < 0 else value


def synthetic_fields(engine_id: int, rng: np.random.Generator) -> dict[str, tuple[int, int]]:
    base = rng.normal()
    cutoff = int(np.clip(16000 + base * 9000, 0, 32767))
    if rng.random() < 0.2:
        cutoff = 32767
    resonance = int(np.clip(16000 - base * 8000 + rng.normal() * 2000, 0, 32767))
    attack = 0 if rng.random() < 0.4 else int(rng.integers(1, 32767))
    mod_active = rng.random() < 0.5
    destination = DESTINATIONS[int(rng.integers(len(DESTINATIONS)))] if mod_active else DESTINATIONS[0]
    fields = {
        "VCO.Type": (ENGINE_COUNT, engine_value(engine_id)),
        "VCO.Param1": (CONTINUOUS_META, int(rng.integers(0, 32768))),
        "VCF.Cutoff": (CONTINUOUS_META, cutoff),
        "VCF.Reso": (CONTINUOUS_META, resonance),
        # EG1 and LFO are sampled as a set, so their fields need shapes a real preset could have
        "EG1.Attack": (CONTINUOUS_META, int(rng.integers(0, 32768))),
        "EG1.Decay": (CONTINUOUS_META, int(rng.integers(0, 32768))),
        "EG2.Attack": (CONTINUOUS_META, attack),
        "LFO.Rate": (CONTINUOUS_META, int(rng.integers(0, 32768))),
        "LFO.Shape": (OCTAVE_META, OCTAVES[int(rng.integers(len(OCTAVES)))]),
        "Kbd.Octave": (OCTAVE_META, OCTAVES[int(rng.integers(len(OCTAVES)))]),
        "Kbd.Hold": (2, int(rng.random() < 0.3)),
        "Arp.Enable": (2, int(rng.random() < 0.3)),
        "Arp.SeqOn": (2, 0),
        "Co1.EG1": (0, signed(0 if rng.random() < 0.7 else int(base * 15000))),
        "Co5.LFO": (0, signed(int(rng.integers(2000, 20000)) if mod_active else 0)),
        "Mat.Assign1": (0, int(destination[0])),
        "Mat.Assign2": (0, int(destination[1])),
        "Mat.Assign3": (0, int(destination[2])),
    }
    fields.update({f"Pad.F{i:02d}": (0, 7) for i in range(FILLER_FIELDS)})
    return fields


def synthetic_preset(
    engine_id: int, index: int, rng: np.random.Generator, category: int | None = None
) -> Preset:
    if category is None:
        category = (0, 3, 5)[index % 3]
    return Preset(
        name=f"Syn {engine_id} {index:03d}",
        category=category,
        body=build_body(synthetic_fields(engine_id, rng)),
    )


def write_training_dir(
    root: Path, presets_per_engine: int = 60, engines: tuple[int, ...] = (1, 3), seed: int = 0
) -> Path:
    """A training directory with a folder of loose presets and a bank per engine."""
    rng = np.random.default_rng(seed)
    for engine_id in engines:
        presets = [synthetic_preset(engine_id, i, rng) for i in range(presets_per_engine)]
        half = presets_per_engine // 2
        loose = root / f"Engine{engine_id}"
        loose.mkdir(parents=True, exist_ok=True)
        for i, preset in enumerate(presets[:half]):
            (loose / f"{i:03d}.mbp").write_bytes(preset.to_bytes())
        (root / f"engine{engine_id}.mfprojz").write_bytes(write_bank(presets[half:], f"Engine {engine_id}"))
    return root


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()
