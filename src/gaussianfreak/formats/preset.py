"""Single preset files (.mfp, and .mbp inside banks).

Text layout (Boost serialization archive)::

    22 serialization::archive 10 0 4 <len> <version> <len> <name> <category> 0 0
    18 <characteristics> <init> 0 <p1> <datalen> <signed body bytes...>

Strings are length-prefixed and read by character count, so names may contain spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from gaussianfreak.formats.constants import MAX_NAME_LEN

_TOKEN = re.compile(rb"\s*(\S+)")
_NO_CHARACTERISTICS = "0" * 18


@dataclass(frozen=True, slots=True)
class Preset:
    """A parsed preset file. Serialising an unmodified parse reproduces the file byte for byte."""

    name: str
    category: int
    body: bytes
    version: str = "174"
    characteristics: str = _NO_CHARACTERISTICS
    init: int = 0
    p1: int = 0
    archive_header: tuple[str, str, str] = ("10", "0", "4")
    reserved: tuple[str, str, str] = ("0", "0", "0")  # the two zeros after category, the zero after init
    trailer: bytes = b"\n"

    @classmethod
    def from_bytes(cls, raw: bytes) -> Preset:
        return _Parser(raw).parse()

    @classmethod
    def empty_slot(cls) -> Preset:
        """An empty (Init) slot, as MIDI Control Center writes them inside banks."""
        return cls(name="Init", category=0, body=b"", version="207", init=1, p1=51)

    def to_bytes(self) -> bytes:
        parts = [
            "22 serialization::archive",
            *self.archive_header,
            str(len(self.version)),
            self.version,
            str(len(self.name)),
        ]
        if self.name:
            parts.append(self.name)
        parts += [
            str(self.category),
            self.reserved[0],
            self.reserved[1],
            str(len(self.characteristics)),
            self.characteristics,
            str(self.init),
            self.reserved[2],
            str(self.p1),
            str(len(self.body)),
        ]
        parts += [str(b - 256 if b > 127 else b) for b in self.body]
        return " ".join(parts).encode("latin-1") + self.trailer

    def renamed(self, name: str, category: int) -> Preset:
        """Copy with a device-safe name and new category, characteristics cleared."""
        return replace(
            self, name=safe_name(name), category=category, characteristics=_NO_CHARACTERISTICS, init=0
        )

    def with_body(self, body: bytes) -> Preset:
        return replace(self, body=body)


def safe_name(name: str) -> str:
    """Device-safe preset name: up to 14 characters from the MicroFreak alphabet."""
    cleaned = re.sub(r"[^ A-Za-z0-9._-]", "_", name)[:MAX_NAME_LEN]
    return cleaned or "Untitled"


class _Parser:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self._pos = 0

    def parse(self) -> Preset:
        if self._token() != b"22" or self._token() != b"serialization::archive":
            raise ValueError("not a MicroFreak preset (serialization::archive header missing)")
        archive_header = (self._token().decode(), self._token().decode(), self._token().decode())
        version = self._string()
        name = self._string()
        category = int(self._token())
        reserved_a, reserved_b = self._token().decode(), self._token().decode()
        characteristics = self._string()
        init = int(self._token())
        reserved_c = self._token().decode()
        p1 = int(self._token())
        datalen = int(self._token())
        body = bytes(int(self._token()) & 0xFF for _ in range(datalen))
        return Preset(
            name=name,
            category=category,
            body=body,
            version=version,
            characteristics=characteristics,
            init=init,
            p1=p1,
            archive_header=archive_header,
            reserved=(reserved_a, reserved_b, reserved_c),
            trailer=self._raw[self._pos :],
        )

    def _token(self) -> bytes:
        match = _TOKEN.match(self._raw, self._pos)
        if not match:
            raise ValueError("unexpected end of preset file")
        self._pos = match.end()
        return match.group(1)

    def _string(self) -> str:
        length = int(self._token())
        if length == 0:
            return ""
        start = self._pos + 1
        self._pos = start + length
        return self._raw[start : self._pos].decode("latin-1")
