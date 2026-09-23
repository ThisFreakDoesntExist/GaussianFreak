# Training data

Put the MicroFreak presets the model should learn from in this folder. **Preset files are not part of the
repository**: everything here except this README is git-ignored.

Only train on presets you are allowed to use: your own sounds, presets that came with your MicroFreak, or presets
whose terms allow it.

## Layout

Any layout works. Every file below this folder is scanned recursively, and zip containers are read in place,
including zips nested inside zips; don't unpack them. Other files (PDFs, images, samples) are skipped.

## File formats

| Extension | What it is |
|---|---|
| `.mfp` | Single preset exported by MIDI Control Center |
| `.mfpz` | Zipped single preset, one member named `0_<preset name>`: what MCC's preset import takes |
| `.mbp` | Single preset inside a bank (MCC's local library stores these as loose files) |
| `.mfprojz` | MCC project: a zip holding `<Bank>/01-<Bank>-A/NN-<name>.mbp`, usually 512 slots |
| `.mfbz` | MCC bank export (zip of presets) |
| `.zip` | Any zip of the above |

Wavetable and sample files (`.mfw`, `.mfwz`, `.mfwbz`, `.mfs`) are never read as presets.

A preset file is a Boost serialization text archive. Its header holds the name, category and characteristics,
followed by a 4,672-byte body written as signed decimal bytes:

```
22 serialization::archive 10 0 4 <len> <version> <len> <name> <category> 0 0 18 <characteristics> <init> 0 <p1> 4672 <body bytes...>
```

The body is MIDI 8-to-7 bit packed. Unpacked (4,088 bytes), it starts with 96–110 self-describing tagged fields
such as `VCF.Cutoff` or `Mat.Assign1`, each with a metadata byte and a 16-bit value. Presets saved by older firmware
have fewer fields; the model treats a missing field as that engine's most common value.

## What gets trained on

`gaussianfreak train` prints what the scan found. The scan:

1. reads every preset, skipping empty (Init) bank slots;
2. treats byte-identical bodies as one preset, however many banks it appears in;
3. treats presets with the same engine and the same sound-field values as one sound. They may still differ in
   arpeggiator, sequencer, key root/scale or hold. The copy saved by the newest firmware is kept;
4. excludes presets that
   - have a truncated body (fewer than 96 tagged fields),
   - use the WaveUser, Sample or Grains engines (they need user wavetables or samples), or
   - have an undecodable engine.

More presets make a better model; engines with only a few presets borrow correlation structure from the rest.
