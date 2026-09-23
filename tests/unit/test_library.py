from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaussianfreak.dataset import Exclusion, PresetLibrary, PresetRecord
from gaussianfreak.formats import Preset, read_fields, write_bank, write_fields
from tests.synthetic import build_body, engine_value, synthetic_fields, synthetic_preset


def test_every_synthetic_preset_is_training_ready(records: list[PresetRecord]) -> None:
    assert len(records) == 120
    assert {r.engine_id for r in records} == {1, 3}
    assert [r.name.lower() for r in records] == sorted(r.name.lower() for r in records)


def test_missing_directory() -> None:
    with pytest.raises(FileNotFoundError):
        PresetLibrary(Path("/nonexistent/training")).scan()


class TestDeduplication:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(1)
        original = synthetic_preset(3, 1, rng, category=5)
        # same body in a bank under another name: one preset, named by the first copy by path
        copy = Preset(name="Renamed copy", category=3, body=original.body)
        # same sound, different arpeggiator setting: a sound duplicate
        arp = write_fields(original.body, {"Arp.Enable": 1 - read_fields(original.body)["Arp.Enable"].value})
        # different octave: a different sound
        octave = write_fields(
            original.body,
            {"Kbd.Octave": 21845 if read_fields(original.body)["Kbd.Octave"].value != 21845 else 5461},
        )
        truncated = Preset(name="Truncated", category=0, body=build_body({"VCO.Type": (22, engine_value(3))}))
        sampler = synthetic_fields(19, rng)
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "Pack.mfp").write_bytes(original.to_bytes())
        (tmp_path / "b").mkdir()
        bank = [
            copy,
            Preset(name="Tweaked arp", category=5, body=arp),
            Preset(name="Octave variant", category=5, body=octave),
            truncated,
            Preset(name="Sampler", category=0, body=build_body(sampler)),
        ]
        (tmp_path / "b" / "Bank.mfprojz").write_bytes(write_bank(bank, "Bank"))
        (tmp_path / "b" / "notes.txt").write_text("not a preset")
        return tmp_path

    def test_scan(self, root: Path) -> None:
        scan = PresetLibrary(root).scan()
        by_name = {e.name: e for e in scan.entries}
        assert scan.preset_copies == 6
        assert scan.empty_slots == 512 - 5
        assert set(by_name) == {"Syn 3 001", "Tweaked arp", "Octave variant", "Truncated", "Sampler"}
        assert by_name["Syn 3 001"].record is not None
        assert by_name["Syn 3 001"].record.category == 5
        assert by_name["Tweaked arp"].exclusions == (Exclusion.SOUND_DUPLICATE,)
        assert by_name["Octave variant"].trainable
        assert Exclusion.TRUNCATED_FIELDS in by_name["Truncated"].exclusions
        assert by_name["Sampler"].exclusions == (Exclusion.NEEDS_USER_CONTENT,)
        assert {r.name for r in scan.records} == {"Syn 3 001", "Octave variant"}
        assert scan.exclusion_counts()[Exclusion.SOUND_DUPLICATE] == 1
