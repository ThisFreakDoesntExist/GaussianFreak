from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaussianfreak.fields import PERFORMANCE_DEFAULTS, is_sound_field
from gaussianfreak.formats import BODY_LEN, CATEGORIES, Preset, engine_of, iter_preset_files, read_fields
from gaussianfreak.generate import GenerationError, PresetGenerator, export_bank, near_copy_threshold
from gaussianfreak.model import CopulaPresetModel
from gaussianfreak.model.template import init_template


@pytest.fixture(scope="module")
def generator(model: CopulaPresetModel) -> PresetGenerator:
    return PresetGenerator(model)


def test_generated_presets_are_valid_files_of_their_engine(generator: PresetGenerator) -> None:
    for g in generator.generate(10, np.random.default_rng(0), engine_id=3):
        parsed = Preset.from_bytes(g.preset.to_bytes())
        assert engine_of(read_fields(parsed.body)) == 3
        assert len(parsed.body) == BODY_LEN
        assert parsed.category in range(len(CATEGORIES))
        assert g.filename == f"{parsed.name}.mfpz"


def test_generated_presets_play_as_plain_sounds(generator: PresetGenerator) -> None:
    for g in generator.generate(10, np.random.default_rng(1), engine_id=1):
        fields = read_fields(g.preset.body)
        for key, value in PERFORMANCE_DEFAULTS.items():
            assert fields[key].value == value


def test_every_engine_writes_into_the_init_preset(model: CopulaPresetModel) -> None:
    template = init_template()
    assert template.name == "Init"
    assert all(em.template is template for em in model.engines.values())


def test_non_sound_fields_come_from_the_init_preset(generator: PresetGenerator) -> None:
    init = init_template().fields
    for engine_id in (1, 3):
        for g in generator.generate(5, np.random.default_rng(2), engine_id=engine_id):
            fields = read_fields(g.preset.body)
            assert list(fields) == list(init)
            assert engine_of(fields) == engine_id
            for key, value in fields.items():
                if not is_sound_field(key) and key not in PERFORMANCE_DEFAULTS and key != "VCO.Type":
                    assert value == init[key], key


def test_near_copies_are_rejected(generator: PresetGenerator, model: CopulaPresetModel) -> None:
    threshold = near_copy_threshold(model.engine(3))
    assert all(g.distance >= threshold for g in generator.generate(20, np.random.default_rng(3), engine_id=3))


def test_same_seed_same_presets(generator: PresetGenerator) -> None:
    a = generator.generate(4, np.random.default_rng(7))
    b = generator.generate(4, np.random.default_rng(7))
    assert [g.preset.to_bytes() for g in a] == [g.preset.to_bytes() for g in b]


def test_mixed_generation_numbers_presets_across_engines(generator: PresetGenerator) -> None:
    generated = generator.generate(12, np.random.default_rng(4))
    assert [int(g.preset.name[-3:]) for g in generated] == list(range(1, 13))


def test_category_is_honoured_or_falls_back(generator: PresetGenerator) -> None:
    pads = generator.generate(5, np.random.default_rng(5), engine_id=3, category=CATEGORIES.index("Pad"))
    assert {g.category_name for g in pads} == {"Pad"}
    strings = generator.generate(
        5, np.random.default_rng(5), engine_id=3, category=CATEGORIES.index("Strings")
    )
    assert {g.category_name for g in strings} <= {"Bass", "Lead", "Pad"}


def test_unknown_engine(generator: PresetGenerator) -> None:
    with pytest.raises(GenerationError):
        generator.generate(1, np.random.default_rng(0), engine_id=9)


def test_export_bank(generator: PresetGenerator, tmp_path: Path) -> None:
    generated = generator.generate(3, np.random.default_rng(6), engine_id=1)
    export = export_bank(generated, tmp_path, "My Bank")
    assert sorted(p.name for p in export.presets_dir.iterdir()) == sorted(g.filename for g in generated)
    # each single preset is a .mfpz holding the preset itself, which MIDI Control Center imports
    for g in generated:
        path = export.presets_dir / g.filename
        [(_, raw)] = list(iter_preset_files(path.name, path.read_bytes()))
        assert Preset.from_bytes(raw) == g.preset
    assert export.bank.stat().st_size > 0
    assert len(export.report.read_text().splitlines()) == 4
