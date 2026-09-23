from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pytest

from gaussianfreak.formats import (
    BODY_LEN,
    UNPACKED_LEN,
    Preset,
    engine_of,
    engine_type_value,
    iter_preset_files,
    pack_7bit,
    read_fields,
    unpack_7bit,
    write_bank,
    write_fields,
    write_preset_archive,
)
from gaussianfreak.formats.bank import ENTRY_TIMESTAMP
from tests.synthetic import build_body, synthetic_preset, zip_bytes


@pytest.fixture
def preset() -> Preset:
    return synthetic_preset(3, 0, np.random.default_rng(0))


class TestPacking:
    def test_round_trip(self) -> None:
        data = bytes(range(256)) * 14 + bytes(UNPACKED_LEN - 256 * 14)
        assert unpack_7bit(pack_7bit(data)) == data
        assert len(pack_7bit(data)) == BODY_LEN

    def test_rejects_partial_blocks(self) -> None:
        with pytest.raises(ValueError, match="multiple of 7"):
            pack_7bit(b"123")


class TestPresetText:
    def test_serialize_round_trips(self, preset: Preset) -> None:
        raw = preset.to_bytes()
        assert Preset.from_bytes(raw) == preset
        assert Preset.from_bytes(raw).to_bytes() == raw

    def test_names_with_double_spaces_survive(self) -> None:
        preset = Preset(name="93s  Rave", category=0, body=b"")
        assert Preset.from_bytes(preset.to_bytes()).name == "93s  Rave"

    def test_empty_slot(self) -> None:
        slot = Preset.from_bytes(Preset.empty_slot().to_bytes())
        assert (slot.name, slot.init, slot.body) == ("Init", 1, b"")

    def test_rejects_other_files(self) -> None:
        with pytest.raises(ValueError, match="not a MicroFreak preset"):
            Preset.from_bytes(b"hello world")

    def test_renamed_is_device_safe(self, preset: Preset) -> None:
        renamed = preset.renamed("A very long name/with*symbols", 5)
        assert renamed.name == "A very long na"
        assert renamed.category == 5
        assert renamed.body == preset.body


class TestTaggedFields:
    def test_reads_fields(self, preset: Preset) -> None:
        fields = read_fields(preset.body)
        assert engine_of(fields) == 3
        assert "VCF.Cutoff" in fields

    def test_write_changes_only_the_target_value(self, preset: Preset) -> None:
        before = read_fields(preset.body)
        new_value = (before["VCF.Cutoff"].value + 1234) % 32768
        after = read_fields(write_fields(preset.body, {"VCF.Cutoff": new_value}))
        assert after["VCF.Cutoff"].value == new_value
        assert {k: v for k, v in after.items() if k != "VCF.Cutoff"} == {
            k: v for k, v in before.items() if k != "VCF.Cutoff"
        }

    def test_write_rejects_unknown_fields_and_out_of_range_values(self, preset: Preset) -> None:
        with pytest.raises(KeyError):
            write_fields(preset.body, {"VCF.NotAField": 1})
        with pytest.raises(ValueError, match="uint16"):
            write_fields(preset.body, {"VCF.Cutoff": 70000})

    @pytest.mark.parametrize("metadata", [17, 18, 22])
    def test_engine_type_value_round_trips(self, metadata: int) -> None:
        for engine_id in range(1, metadata + 1):
            body = build_body({"VCO.Type": (metadata, engine_type_value(engine_id, metadata))})
            assert engine_of(read_fields(body)) == engine_id

    def test_engine_type_value_rejects_unaddressable_engines(self) -> None:
        with pytest.raises(ValueError, match="not addressable"):
            engine_type_value(19, 18)

    def test_unknown_engine(self) -> None:
        assert engine_of({}) is None
        assert engine_of(read_fields(build_body({"VCO.Type": (0, 5)}))) is None


class TestBank:
    def test_mcc_layout(self, preset: Preset) -> None:
        data = write_bank([preset], "Test Bank")
        archive = zipfile.ZipFile(io.BytesIO(data))
        names = archive.namelist()
        assert len(names) == 514
        assert names[:2] == ["Test Bank/", "Test Bank/01-Test Bank-A/"]
        folder = "Test Bank/01-Test Bank-A/"
        assert Preset.from_bytes(archive.read(f"{folder}01-Test Bank-A1.mbp")).name == preset.name
        assert Preset.from_bytes(archive.read(f"{folder}512-Test Bank-A512.mbp")).init == 1

    def test_every_slot_file_carries_its_slot_like_arturia_banks(self) -> None:
        rng = np.random.default_rng(1)
        presets = [synthetic_preset(1, i, rng).renamed(f"Preset {i}", 0) for i in range(1, 35)]
        archive = zipfile.ZipFile(io.BytesIO(write_bank(presets, "Order")))
        files = [n for n in archive.namelist() if n.endswith(".mbp")]
        assert files == sorted(files)
        slot_names = {}
        for name in files:
            match = re.fullmatch(r"Order/01-Order-A/(\d+)-Order-A(\d+)\.mbp", name)
            assert match, name
            assert match.group(1).lstrip("0") == match.group(2)
            slot_names[int(match.group(2))] = Preset.from_bytes(archive.read(name)).name
        assert sorted(slot_names) == list(range(1, 513))
        assert [slot_names[s] for s in range(1, 35)] == [p.name for p in presets]
        assert all(slot_names[s] == "Init" for s in range(35, 513))

    def test_a_preset_archive_holds_one_member_named_like_mcc(self, preset: Preset) -> None:
        """A .mfpz is a zip of one member, ``0_<preset name>``, with no extension - MCC's own layout."""
        archive = zipfile.ZipFile(io.BytesIO(write_preset_archive(preset)))
        assert archive.namelist() == [f"0_{preset.name}"]
        assert Preset.from_bytes(archive.read(f"0_{preset.name}")).name == preset.name

    def test_a_preset_archive_is_read_back_like_a_real_one(self, preset: Preset) -> None:
        """The library scanner finds presets in real .mfpz files by sniffing; ours must look the same to it."""
        data = write_preset_archive(preset)
        members = list(iter_preset_files("Downloaded.mfpz", data))
        assert [Preset.from_bytes(raw).name for _, raw in members] == [preset.name]

    def test_archives_of_the_same_preset_are_byte_identical(self, preset: Preset) -> None:
        """A download has to reproduce byte for byte for a seed; a zip entry's timestamp would drift."""
        archive = zipfile.ZipFile(io.BytesIO(write_preset_archive(preset)))
        assert {info.date_time for info in archive.infolist()} == {ENTRY_TIMESTAMP}
        assert write_preset_archive(preset) == write_preset_archive(preset)

    def test_two_banks_of_the_same_presets_are_byte_identical(self, preset: Preset) -> None:
        """The API serves a bank per request, and the same seed has to give the same bytes; a zip entry's own
        timestamp would otherwise move it along with the clock."""
        archive = zipfile.ZipFile(io.BytesIO(write_bank([preset], "Same")))
        assert {info.date_time for info in archive.infolist()} == {ENTRY_TIMESTAMP}
        assert write_bank([preset], "Same") == write_bank([preset], "Same")

    def test_too_many_presets(self, preset: Preset) -> None:
        with pytest.raises(ValueError, match="do not fit"):
            write_bank([preset] * 3, "Small", slots=2)


class TestContainers:
    def test_finds_presets_in_nested_zips(self, preset: Preset) -> None:
        raw = preset.to_bytes()
        inner = zip_bytes({"0_My Preset": raw, "wave.mfw": b"22 serialization::archive not a preset"})
        outer = zip_bytes(
            {
                "pack/a.mfp": raw,
                "pack/inner.mfpz": inner,
                "__MACOSX/pack/._a.mfp": b"junk",
                "readme.pdf": b"%PDF",
            }
        )
        found = list(iter_preset_files("pack.zip", outer))
        assert [label for label, _ in found] == [
            "pack.zip!pack/a.mfp",
            "pack.zip!pack/inner.mfpz!0_My Preset",
        ]
        assert all(data == raw for _, data in found)

    def test_plain_files_and_bad_zips(self, preset: Preset) -> None:
        assert list(iter_preset_files("a.mfp", preset.to_bytes())) == [("a.mfp", preset.to_bytes())]
        assert list(iter_preset_files("broken.mfprojz", b"not a zip")) == []
        assert list(iter_preset_files("notes.txt", b"text")) == []
