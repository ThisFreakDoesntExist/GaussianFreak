"""Find preset files inside plain files and (nested) zip containers."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator

PRESET_EXTENSIONS = (".mfp", ".mbp")
CONTAINER_EXTENSIONS = (".zip", ".mfpz", ".mfprojz", ".mfbz")
# Wavetable and sample banks: containers, but never presets.
_CONTENT_EXTENSIONS = (".mfw", ".mfs", ".mfwz", ".mfwbz")
_PRESET_MAGIC = b"22 serialization::archive"
SUPPORTED_EXTENSIONS = PRESET_EXTENSIONS + CONTAINER_EXTENSIONS


def iter_preset_files(label: str, data: bytes) -> Iterator[tuple[str, bytes]]:
    """Yield ``(member_label, raw_preset_bytes)`` for every preset in ``data``, recursing into zips.

    Member labels read ``outer.zip!inner/path.mbp``.
    """
    lower = label.lower()
    if data[:2] == b"PK" or lower.endswith(CONTAINER_EXTENSIONS):
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            return
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or "__MACOSX/" in name or name.split("/")[-1].startswith("._"):
                continue
            inner = archive.read(info)
            inner_label = f"{label}!{name}"
            lname = name.lower()
            # .mfpz members have no extension (MCC names them "0_<preset name>"): sniff the header.
            sniffed = inner.startswith(_PRESET_MAGIC) and not lname.endswith(
                CONTAINER_EXTENSIONS + _CONTENT_EXTENSIONS
            )
            if lname.endswith(PRESET_EXTENSIONS) or sniffed:
                yield inner_label, inner
            elif inner[:2] == b"PK" or lname.endswith(CONTAINER_EXTENSIONS):
                yield from iter_preset_files(inner_label, inner)
    elif lower.endswith(PRESET_EXTENSIONS):
        yield label, data
