"""The Init preset every generated preset is written into.

``init.mfp`` is the MicroFreak's Init sound as exported from the device (``formats/Init.mfpz`` in
github.com/feakk/Microfreak). MIDI Control Center only stores Init slots as empty placeholders, so this is the one
Init with a full preset body. Its tagged-field layout holds every field the model samples on every engine; the engine
itself is set per preset in ``VCO.Type``.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

from gaussianfreak.formats import Preset, read_fields
from gaussianfreak.model.engine_model import TemplatePreset

INIT_RESOURCE = "init.mfp"


@cache
def init_template() -> TemplatePreset:
    preset = Preset.from_bytes(files("gaussianfreak.model").joinpath(INIT_RESOURCE).read_bytes())
    return TemplatePreset(name=preset.name, preset=preset, fields=read_fields(preset.body))
