# GaussianFreak

Microfreak presets autogened through stats (sorry no deep learning). They probably don't exist.

GaussianFreak learns from your own Arturia MicroFreak presets how each parameter is distributed and how parameters
move together, then samples new presets that are neither near-copies of real ones nor random knob settings. No
presets or trained models ship with this repository: bring your own presets, train your own model.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 if needed.

```sh
uv sync
# put preset files you exported from MIDI Control Center under training/ (see training/README.md)
uv run gaussianfreak train                                       # -> artifacts/copula-model.pkl (about 20 s)
uv run gaussianfreak generate --engine Wavetable --category Pad --count 8   # -> generated/
```

`generate` writes `generated/<bank>/*.mfpz` (single presets MIDI Control Center imports), `generated/<bank>.mfprojz`
(a 512-slot bank) and `generated/<bank>.csv`, which lists each preset's engine, category and nearest real preset.
Leave out `--engine` to mix engines; `--seed`, `--bank` and `--out` are optional. `gaussianfreak --help` lists
everything.

The trained model is a Python pickle. Only load models you trained yourself.

## How the model works

1. **Data** (`dataset.py`). Every preset under `training/` is read, including presets inside zips and banks.
   Byte-identical copies count once, and so do presets that differ only in arpeggiator, sequencer or key
   mapping. Presets that need user samples or wavetables are dropped.
2. **Fields** (`fields.py`). Only sound fields are modelled, including keyboard octave. Arpeggiator, sequencer, key
   mapping, system fields, sample references and the engine itself are not. A field is modelled on an engine
   when at least half its presets have it, or at least 30 do. Each field is one of:
   - `continuous`, a knob;
   - `discrete`, snapping to observed values;
   - `bipolar`, a mod amount;
   - `categorical`, a mod destination.
3. **Model** (`model/`), one per oscillator engine:
   - every field keeps its **exact real distribution**, including point masses such as unused mod routes and
     knobs parked at 0% or 100%. Sampling never interpolates out of a point mass;
   - a **Gaussian copula** carries how fields move together. It is calibrated so that simulated rank
     correlations match the real ones;
   - engines with few presets **borrow correlation** from a copula pooled over all engines, with weight
     `50 / (50 + n)`. Oscillator knobs are the exception, because their meaning is engine-specific;
   - categories with at least 12 presets on an engine get their own field distributions;
   - the envelope (EG1) and LFO are taken as a set from one real preset of the same engine and category, because
     sampling their fields one by one loses the shapes real presets use;
   - assignable mod destinations are labels, not amounts. They stay out of the copula, and each sample takes a
     whole real destination trio from presets whose assignable mod columns are active in the same pattern.
4. **Generation** (`generate.py`). Every preset starts from the MicroFreak's Init preset (`model/init.mfp`, the
   device default as exported in github.com/feakk/Microfreak), whatever its engine. The engine is set in
   `VCO.Type` and the sampled values are written in. Every other sound field gets the engine's most common value
   in the training presets, so nothing of Init's own sound remains where real presets differ. The arpeggiator,
   sequencer and hold are switched off. A sample closer to a real preset than half that engine's 5th-percentile
   real gap is rejected as a near-copy.

## Layout

```
src/gaussianfreak/
├── formats/      preset text codec, 7-bit body packing and tagged fields, banks, zip containers
├── dataset.py    training folder scan, deduplication, exclusions
├── fields.py     which fields are sound, and their kinds
├── model/        empirical Gaussian copula, destination sampler, per-engine model, Init preset
├── train.py      scan, fit, save
├── generate.py   sampling, near-copy rejection, bank export
└── cli.py        `gaussianfreak train | generate`
tests/
├── unit/         synthetic presets, no training data needed
└── integration/  real presets (skipped when training/ is empty)
```

## Development

```sh
uv run pytest                          # all tests (integration tests skip without training data)
uv run ruff format src tests && uv run ruff check src tests
uv run mypy
```

## License

MIT. MicroFreak and MIDI Control Center are trademarks of Arturia; this project is not affiliated with Arturia.
