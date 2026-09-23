"""MIDI Control Center archives: banks (.mfprojz) and single presets (.mfpz)."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Sequence

from gaussianfreak.formats.preset import Preset

BANK_SLOTS = 512
ENTRY_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
"""Fixed date on every entry, so the same seed writes the same archive byte for byte; a zip entry's own timestamp
would otherwise change with the clock."""
PRESET_ENTRY_PREFIX = "0_"
"""How MCC names the single member of a .mfpz: ``0_<preset name>``, with no extension."""


def write_bank(presets: Sequence[Preset], bank_name: str, slots: int = BANK_SLOTS) -> bytes:
    """Build an MCC-style .mfprojz: ``<Bank>/01-<Bank>-A/<NN>-<Bank>-A<slot>.mbp`` for every slot, Init slots last.

    Slot files are named the way Arturia's own banks name them. MIDI Control Center takes the slot from the
    ``-A<slot>`` suffix; without it, slots are ordered by filename text (1-10, 100-109, 11, ...).
    """
    if len(presets) > slots:
        raise ValueError(f"{len(presets)} presets do not fit in {slots} slots")
    bank = re.sub(r"[^A-Za-z0-9 _-]", "_", bank_name) or "Bank"
    folder = f"{bank}/01-{bank}-A/"
    entries = {}
    for slot in range(1, slots + 1):
        preset = presets[slot - 1] if slot <= len(presets) else Preset.empty_slot()
        entries[f"{folder}{slot:02d}-{bank}-A{slot}.mbp"] = preset.to_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo(f"{bank}/", ENTRY_TIMESTAMP), b"")
        zf.writestr(zipfile.ZipInfo(folder, ENTRY_TIMESTAMP), b"")
        # entries in filename order, as in Arturia's banks
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, ENTRY_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED  # a ZipInfo does not inherit the archive's default
            zf.writestr(info, entries[name])
    return buf.getvalue()


def write_preset_archive(preset: Preset) -> bytes:
    """Build a .mfpz: the zipped single preset that MIDI Control Center imports as a preset.

    MCC exports single presets as a bare ``.mfp`` but will not import one; its preset import takes this zip,
    holding one member named ``0_<preset name>`` with no extension. The name needs no escaping because
    :func:`~gaussianfreak.formats.preset.safe_name` has already limited it to the device's alphabet.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(f"{PRESET_ENTRY_PREFIX}{preset.name}", ENTRY_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED  # a ZipInfo does not inherit the archive's default
        zf.writestr(info, preset.to_bytes())
    return buf.getvalue()
