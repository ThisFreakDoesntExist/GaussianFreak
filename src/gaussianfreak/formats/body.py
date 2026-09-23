"""Preset bodies: MIDI 8-to-7 bit packing and the self-describing tagged fields inside.

Unpacked, a body starts with tagged fields grouped by three-letter section::

    '#' GRP                         first group
    0x40+len <name> 'c' meta lo hi  one field: metadata byte and little-endian uint16 value
    '@' '#' GRP                     next group

Fields are addressed as ``"GRP.Name"``, for example ``"VCF.Cutoff"``.
"""

from __future__ import annotations

from typing import NamedTuple

from gaussianfreak.formats.constants import ENGINES

_GROUP_MARKER = 0x23  # '#'
_NEXT_GROUP = 0x40  # '@'
_NAME_LENGTH_BASE = 0x40
_VALUE_MARKER = 0x63  # 'c'
_MAX_NAME_LENGTH = 31
_UINT16_MAX = 0xFFFF
_FULL_SCALE = 32767


class TaggedField(NamedTuple):
    metadata: int
    value: int


def unpack_7bit(body: bytes) -> bytes:
    out = bytearray()
    for block in range(len(body) // 8):
        base = block * 8
        bitmap = body[base]
        for i in range(7):
            out.append(body[base + 1 + i] | (0x80 if (bitmap >> i) & 1 else 0))
    return bytes(out)


def pack_7bit(data: bytes) -> bytes:
    if len(data) % 7:
        raise ValueError("unpacked data length must be a multiple of 7")
    out = bytearray()
    for block in range(len(data) // 7):
        chunk = data[block * 7 : block * 7 + 7]
        bitmap = 0
        for i, b in enumerate(chunk):
            if b & 0x80:
                bitmap |= 1 << i
        out.append(bitmap)
        out.extend(b & 0x7F for b in chunk)
    return bytes(out)


def _field_offsets(unpacked: bytes) -> dict[str, tuple[int, int]]:
    """``{"GRP.Name": (metadata_offset, value_offset)}`` for the tagged prefix."""
    pos, group, offsets = 0, "", {}
    while pos + 5 <= len(unpacked):
        if pos == 0 and unpacked[0] == _GROUP_MARKER:
            group, pos = unpacked[1:4].decode("latin-1"), 4
            continue
        if unpacked[pos] == _NEXT_GROUP and unpacked[pos + 1] == _GROUP_MARKER:
            group, pos = unpacked[pos + 2 : pos + 5].decode("latin-1"), pos + 5
            continue
        name_len = unpacked[pos] - _NAME_LENGTH_BASE
        end = pos + 1 + name_len
        if (
            not group
            or not 1 <= name_len <= _MAX_NAME_LENGTH
            or end + 4 > len(unpacked)
            or unpacked[end] != _VALUE_MARKER
        ):
            break
        offsets[f"{group}.{unpacked[pos + 1 : end].decode('latin-1')}"] = (end + 1, end + 2)
        pos = end + 4
    return offsets


def read_fields(body: bytes) -> dict[str, TaggedField]:
    """All tagged fields of a body; empty for legacy (untagged) bodies."""
    unpacked = unpack_7bit(body)
    return {
        key: TaggedField(unpacked[m], unpacked[v] | (unpacked[v + 1] << 8))
        for key, (m, v) in _field_offsets(unpacked).items()
    }


def write_fields(body: bytes, values: dict[str, int]) -> bytes:
    """A new body with the given fields' uint16 values replaced in place."""
    unpacked = bytearray(unpack_7bit(body))
    offsets = _field_offsets(bytes(unpacked))
    for key, requested in values.items():
        if key not in offsets:
            raise KeyError(f"field {key!r} not present in this preset layout")
        value = int(requested)
        if not 0 <= value <= _UINT16_MAX:
            raise ValueError(f"{key}: value {value} outside uint16")
        _, v = offsets[key]
        unpacked[v], unpacked[v + 1] = value & 0xFF, value >> 8
    return pack_7bit(bytes(unpacked))


def engine_of(fields: dict[str, TaggedField]) -> int | None:
    """Oscillator engine id encoded in ``VCO.Type``, or None when absent or unknown."""
    if "VCO.Type" not in fields:
        return None
    meta, raw = fields["VCO.Type"]
    if not meta:
        return None
    engine_id = round(raw * meta / _FULL_SCALE)
    return engine_id if 0 < engine_id < len(ENGINES) else None


def engine_type_value(engine_id: int, metadata: int) -> int:
    """Raw ``VCO.Type`` value selecting ``engine_id`` in a body whose ``VCO.Type`` metadata is ``metadata``."""
    if not 0 < engine_id <= metadata:
        raise ValueError(f"engine {engine_id} is not addressable with VCO.Type metadata {metadata}")
    return round(engine_id * _FULL_SCALE / metadata)


def to_signed(value: int) -> int:
    return value - 0x10000 if value >= 0x8000 else value


def from_signed(value: int) -> int:
    return value + 0x10000 if value < 0 else value
