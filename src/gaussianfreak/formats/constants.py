"""MicroFreak firmware constants."""

BODY_LEN = 4672
"""Length of a preset body as stored in files (MIDI 8-to-7 packed)."""
UNPACKED_LEN = 4088
MAX_NAME_LEN = 14

ENGINES: tuple[str | None, ...] = (
    None,
    "BasicWaves",
    "SuperWave",
    "Wavetable",
    "Harmo",
    "KarplusStr",
    "V.Analog",
    "Waveshaper",
    "Two Op. FM",
    "Formant",
    "Chords",
    "Speech",
    "Modal",
    "Noise",
    "Vocoder",
    "Bass",
    "SawX",
    "Harm",
    "WaveUser",
    "Sample",
    "Scan Grains",
    "Cloud Grains",
    "Hit Grains",
)
"""Oscillator engine names by engine id (id 0 is unused)."""

FIRST_USER_CONTENT_ENGINE = 18
"""Engines from this id on (WaveUser, Sample, Grains) play user wavetables or samples."""

CATEGORIES: tuple[str, ...] = (
    "Bass",
    "Brass",
    "Keys",
    "Lead",
    "Organ",
    "Pad",
    "Percussion",
    "Sequence",
    "SFX",
    "Strings",
    "Template",
    "Vocoder",
)
"""Preset categories by category id."""
