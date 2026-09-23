"""Scan the training directory into deduplicated, training-ready preset records.

Every preset file (and every preset inside zip containers) anywhere under the training directory is read. Copies
with byte-identical bodies become one preset; presets whose sound fields are all identical (they differ only in
arpeggiator, sequencer or key mapping) become one sound, keeping the copy saved by the newest firmware.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from gaussianfreak.fields import is_performance_field
from gaussianfreak.formats import BODY_LEN, CATEGORIES, ENGINES, Preset, TaggedField, engine_of, read_fields
from gaussianfreak.formats.constants import FIRST_USER_CONTENT_ENGINE
from gaussianfreak.formats.containers import SUPPORTED_EXTENSIONS, iter_preset_files

logger = logging.getLogger(__name__)

MIN_TAGGED_FIELDS = 96
"""Firmware layouts have 96-110 tagged fields; fewer means the body stopped parsing early."""


class Exclusion(StrEnum):
    SOUND_DUPLICATE = "sound_duplicate"
    """Same engine and sound as another preset; differs only in how the keyboard plays."""
    TRUNCATED_FIELDS = "truncated_fields"
    """Body stopped parsing early."""
    NEEDS_USER_CONTENT = "needs_user_content"
    """Plays user wavetables or samples (WaveUser, Sample, Grains)."""
    UNKNOWN_ENGINE = "unknown_engine"


@dataclass(frozen=True, slots=True)
class PresetRecord:
    """One unique preset body, as the model learns from it."""

    sha1: str
    name: str
    category: int | None
    engine_id: int
    preset: Preset
    fields: dict[str, TaggedField]

    @property
    def engine(self) -> str:
        return ENGINES[self.engine_id] or f"engine {self.engine_id}"

    @property
    def category_name(self) -> str | None:
        return CATEGORIES[self.category] if self.category is not None else None


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    """A unique preset found in the library, with any reasons it is excluded from training."""

    name: str
    exclusions: tuple[Exclusion, ...]
    record: PresetRecord | None
    """The training record; None when the engine is unknown."""

    @property
    def trainable(self) -> bool:
        return not self.exclusions and self.record is not None


@dataclass(frozen=True, slots=True)
class LibraryScan:
    entries: tuple[LibraryEntry, ...]
    preset_copies: int
    empty_slots: int
    parse_errors: tuple[tuple[str, str], ...]

    @property
    def records(self) -> list[PresetRecord]:
        """Training-ready presets, ordered by name."""
        return [e.record for e in self.entries if e.trainable and e.record is not None]

    def exclusion_counts(self) -> Counter[Exclusion]:
        return Counter(reason for e in self.entries for reason in e.exclusions)


class PresetLibrary:
    """The preset files under a training directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def scan(self) -> LibraryScan:
        if not self.root.is_dir():
            raise FileNotFoundError(f"training directory {self.root} does not exist")
        # sha1 -> (preference key, preset): a named copy wins over an unnamed one, then the first by path
        best: dict[str, tuple[tuple[bool, str], Preset]] = {}
        copies = empty = 0
        errors: list[tuple[str, str]] = []
        for label, raw in self._iter_raw_presets():
            try:
                preset = Preset.from_bytes(raw)
            except ValueError as exc:
                errors.append((label, str(exc)))
                continue
            if not preset.body:
                empty += 1
                continue
            if len(preset.body) != BODY_LEN:
                errors.append((label, f"unexpected body length {len(preset.body)}"))
                continue
            copies += 1
            sha1 = hashlib.sha1(preset.body).hexdigest()
            key = (preset.name == "", label)
            if sha1 not in best or key < best[sha1][0]:
                best[sha1] = (key, preset)
        entries = _entries({sha1: preset for sha1, (_, preset) in best.items()})
        logger.info(
            "scanned %d preset copies: %d unique, %d training-ready, %d parse errors",
            copies,
            len(entries),
            sum(e.trainable for e in entries),
            len(errors),
        )
        return LibraryScan(
            entries=tuple(entries), preset_copies=copies, empty_slots=empty, parse_errors=tuple(errors)
        )

    def _iter_raw_presets(self) -> Iterator[tuple[str, bytes]]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or not path.name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            yield from iter_preset_files(str(path.relative_to(self.root)), path.read_bytes())


def _entries(presets: dict[str, Preset]) -> list[LibraryEntry]:
    ordered = sorted(presets.items(), key=lambda kv: (kv[1].name.lower(), kv[0]))
    fields_by_sha = {sha1: read_fields(p.body) for sha1, p in ordered}
    duplicates = _sound_duplicates([(sha1, fields_by_sha[sha1]) for sha1, _ in ordered])

    entries = []
    for sha1, preset in ordered:
        fields = fields_by_sha[sha1]
        engine_id = engine_of(fields)
        exclusions = []
        if sha1 in duplicates:
            exclusions.append(Exclusion.SOUND_DUPLICATE)
        if len(fields) < MIN_TAGGED_FIELDS:
            exclusions.append(Exclusion.TRUNCATED_FIELDS)
        if engine_id is None:
            exclusions.append(Exclusion.UNKNOWN_ENGINE)
        elif engine_id >= FIRST_USER_CONTENT_ENGINE:
            exclusions.append(Exclusion.NEEDS_USER_CONTENT)
        record = None
        if engine_id is not None:
            category = preset.category if 0 <= preset.category < len(CATEGORIES) else None
            record = PresetRecord(
                sha1=sha1,
                name=preset.name,
                category=category,
                engine_id=engine_id,
                preset=preset,
                fields=fields,
            )
        entries.append(LibraryEntry(name=preset.name, exclusions=tuple(exclusions), record=record))
    return entries


def _sound_duplicates(ordered: list[tuple[str, dict[str, TaggedField]]]) -> set[str]:
    """Bodies that repeat another preset's sound.

    Two presets share a sound when they have the same engine and the same value for every sound field that all
    fully parsed presets have. Within a group the copy with the most tagged fields is kept, then the first by name.
    """
    complete = [set(fields) for _, fields in ordered if len(fields) >= MIN_TAGGED_FIELDS]
    shared = set.intersection(*complete) if complete else set()
    core = sorted(k for k in shared if k != "VCO.Type" and not is_performance_field(k))

    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, (sha1, fields) in enumerate(ordered):
        if len(fields) < MIN_TAGGED_FIELDS:
            key: tuple[object, ...] = ("truncated", sha1)
        else:
            key = (engine_of(fields), tuple(fields[k].value for k in core))
        groups[key].append(index)

    duplicates: set[str] = set()
    for members in groups.values():
        keep = max(members, key=lambda i: (len(ordered[i][1]), -i))
        duplicates.update(ordered[i][0] for i in members if i != keep)
    return duplicates
