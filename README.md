# GaussianFreak

Microfreak presets autogened through stats (sorry no deep learning). They probably don't exist.

Give GaussianFreak a folder of MicroFreak presets. It learns correlations between parameters so what comes out of it should mostly sound like what reasonably would be called "musical". Player beware, some really suck. But thats just the roll of the dice

No presets or trained models come with this repository. You bring your own presets and train your own model.

## Inspiration

This project is entirely inspired by [NeuralDX7](https://github.com/Nintorac/NeuralDX7), the model
behind [This DX7 Cart Does Not Exist](https://www.thisdx7cartdoesnotexist.com/). NeuralDX7 trains on a lot more data and uses a VAE on Yamaha DX7 patches. GaussianFreak tries to accomplish the same thing, but with much less data and way more parameters. If you like what you hear, check out the inspiration and don't be afraid to check out the free DX7 synth emulation plugin [Dexed](https://github.com/asb2m10/dexed)

## Getting started

You need [uv](https://docs.astral.sh/uv/), which will fetch Python 3.12 for you if you don't have it.

```sh
uv sync

# 1. Copy presets exported from MIDI Control Center into training/ (see training/README.md).

# 2. Train. Takes about 20 seconds and writes artifacts/copula-model.pkl.
uv run gaussianfreak train

# 3. Generate eight Wavetable pads into generated/.
uv run gaussianfreak generate --engine Wavetable --category Pad --count 8
```

For each run, `generate` writes:

- `generated/<bank>/*.mfpz`: single presets that MIDI Control Center can import;
- `generated/<bank>.mfprojz`: the same presets as a full 512-slot bank;
- `generated/<bank>.csv`: each preset's engine and category, and the real preset it is closest to.

Leave out `--engine` to get a mix of engines. `--seed`, `--bank` and `--out` are optional, and
`uv run gaussianfreak --help` lists everything else.

One warning: the trained model is saved as a Python pickle, and loading a pickle can run code. Only load models
you trained yourself.

## How it works

**Reading the presets** (`dataset.py`). Every preset under `training/` is read, including those inside zips and
banks. Duplicates count once, and so do presets that only differ in their arpeggiator, sequencer or key mapping.
Presets that rely on your own samples or wavetables are left out, since they won't sound right without those
files.

**Choosing what to learn** (`fields.py`). Only the parameters that shape the sound are modelled, keyboard octave
included. The arpeggiator, sequencer, key mapping, system settings and sample references are not, and neither is
the engine, which you choose. Each parameter is treated as one of four kinds:

- a knob (continuous);
- a switch or stepped control that snaps to values real presets use (discrete);
- a mod matrix amount, which can be negative (bipolar);
- a mod destination, which is a label rather than a number (categorical).

**The model** (`model/`). There is one model per oscillator engine.

- Every parameter keeps its exact real distribution. If most presets leave a mod route at zero or park a knob
  at 100%, generated presets do too, instead of getting some value close by.
- A Gaussian copula captures how parameters move together, for example filter cutoff with envelope amount. It is
  tuned so that generated presets show the same correlations as real ones.
- Engines with few presets borrow some correlation structure from all engines combined, with weight
  `50 / (50 + n)` for an engine with `n` presets. The oscillator knobs are the exception, since they mean
  something different on every engine.
- A category with at least 12 presets on an engine (Pads on Wavetable, say) gets its own parameter distributions.
- The envelope and LFO are copied as a set from one real preset of the same engine and category. Drawing their
  settings one at a time produced shapes no one would dial in.
- Assignable mod destinations are copied as a group of three from a real preset whose mod routing looks the
  same.

**Generating** (`generate.py`). Every new preset starts from the MicroFreak's factory Init preset
(`model/init.mfp`, as exported in [feakk/Microfreak](https://github.com/feakk/Microfreak)). GaussianFreak then
sets the engine and writes in the sampled values. Any sound parameter the model doesn't sample gets the value
real presets of that engine use most often, so nothing of the Init sound is left over. The arpeggiator,
sequencer and hold are switched off. If a new preset lands too close to one of your real presets, it is thrown
away and drawn again.

## Project layout

```
src/gaussianfreak/
├── formats/      reading and writing preset files, banks and zip containers
├── dataset.py    scanning the training folder, removing duplicates and unusable presets
├── fields.py     which parameters are sound parameters, and what kind each one is
├── model/        the copula, mod destination sampling, per-engine models, the Init preset
├── train.py      scan, fit and save
├── generate.py   sampling, near-copy rejection, bank export
└── cli.py        the `gaussianfreak train` and `gaussianfreak generate` commands
tests/
├── unit/         run on synthetic presets, so they need no training data
└── integration/  run on whatever is in training/, and skip what your library is too small for
```

## Development

```sh
uv run pytest                                           # integration tests skip when training/ is empty
uv run ruff format src tests && uv run ruff check src tests
uv run mypy
```

## License

MIT. MicroFreak and MIDI Control Center are trademarks of Arturia. This project is not affiliated with Arturia.
