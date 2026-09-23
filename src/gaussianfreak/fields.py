"""Field taxonomy: which tagged fields describe the sound, and what kind of value each holds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from enum import StrEnum


class FieldKind(StrEnum):
    CONTINUOUS = "continuous"
    """Knob position; sampled with interpolation between observed values."""
    DISCRETE = "discrete"
    """Switch or stepped control; snaps to observed values."""
    BIPOLAR = "bipolar"
    """Signed mod-matrix amount."""
    CATEGORICAL = "categorical"
    """Label without order (assignable mod destination)."""


# Change how the keyboard plays (arpeggiator, sequencer, key mapping) or system state, not the sound.
# Kbd.Octave is a sound field: the register is part of the sound (45% of Wavetable presets play one or two
# octaves down).
PERFORMANCE_PREFIXES = ("Arp.", "Seq.", "Sys.")
PERFORMANCE_FIELDS = frozenset({"Kbd.Root", "Kbd.Scale", "Kbd.Hold"})
# Point at user wavetables/samples on the device; meaningless without that content.
SAMPLE_REFERENCE_FIELDS = frozenset({"Gen.SmpHash", "VCO.SmpIdx"})
# Engine identity is chosen explicitly, never sampled.
IDENTITY_FIELDS = frozenset({"VCO.Type"})
# Key mapping that shifts or bends the pitch played; not modelled, but always set to the engine's usual value.
PITCH_MAPPING_FIELDS = frozenset({"Kbd.Scale", "Kbd.Root"})
# Set on every generated preset so it plays as a plain sound.
PERFORMANCE_DEFAULTS: dict[str, int] = {"Arp.Enable": 0, "Arp.SeqOn": 0, "Kbd.Hold": 0}

# Oscillator knobs mean something different on every engine; their correlations are never shared.
ENGINE_SPECIFIC_FIELDS = frozenset({"VCO.Param1", "VCO.Param2", "VCO.Param3"})

# Assignable mod-matrix destinations name a target parameter; each belongs to one mod-matrix column, whose
# amounts only act through that destination.
ASSIGN_COLUMNS: dict[str, str] = {"Mat.Assign1": "Co5", "Mat.Assign2": "Co6", "Mat.Assign3": "Co7"}
CATEGORICAL_FIELDS = frozenset(ASSIGN_COLUMNS)
MOD_AMOUNT_PREFIX = "Co"

MAX_DISCRETE_LEVELS = 12
"""A field with this few distinct values is a switch or stepped control."""
_METADATA_STEPPED_LIMIT = 200


def is_performance_field(key: str) -> bool:
    return key.startswith(PERFORMANCE_PREFIXES) or key in PERFORMANCE_FIELDS


def is_sound_field(key: str) -> bool:
    return not (is_performance_field(key) or key in SAMPLE_REFERENCE_FIELDS or key in IDENTITY_FIELDS)


def is_mod_amount(key: str) -> bool:
    return key.startswith(MOD_AMOUNT_PREFIX)


def classify_field(key: str, metadata_values: Iterable[int], distinct_values: int) -> FieldKind:
    if is_mod_amount(key):
        return FieldKind.BIPOLAR
    if key in CATEGORICAL_FIELDS:
        return FieldKind.CATEGORICAL
    metadata = Counter(metadata_values).most_common(1)[0][0]
    if 1 <= metadata < _METADATA_STEPPED_LIMIT or distinct_values <= MAX_DISCRETE_LEVELS:
        return FieldKind.DISCRETE
    return FieldKind.CONTINUOUS
