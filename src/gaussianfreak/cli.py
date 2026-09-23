"""Command line: ``gaussianfreak train | generate``."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from gaussianfreak.formats import CATEGORIES
from gaussianfreak.formats.bank import BANK_SLOTS
from gaussianfreak.generate import GenerationError, PresetGenerator, export_bank
from gaussianfreak.model import CopulaPresetModel
from gaussianfreak.train import train as train_model

DEFAULT_MODEL = Path("artifacts/copula-model.pkl")

app = typer.Typer(
    help="Train a Gaussian copula model on MicroFreak presets and generate new presets from it.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
ModelPath = Annotated[Path, typer.Option(help="Trained model file")]


@app.callback()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def train(
    training_dir: Annotated[Path, typer.Option(help="Folder of preset files to learn from")] = Path(
        "training"
    ),
    model: ModelPath = DEFAULT_MODEL,
    seed: Annotated[int, typer.Option(help="Seed for correlation calibration")] = 0,
) -> None:
    """Fit the copula model on the presets under the training folder and save it."""
    fitted, scan = train_model(training_dir, model, seed)
    records = scan.records
    typer.echo(
        f"{scan.preset_copies} preset copies, {len(scan.entries)} unique, {len(records)} training-ready"
    )
    for reason, count in sorted(scan.exclusion_counts().items()):
        typer.echo(f"  excluded {reason}: {count}")
    if scan.parse_errors:
        typer.echo(f"  unreadable files: {len(scan.parse_errors)}")
    typer.echo(f"\n{'Engine':12s} {'Presets':>7s}")
    for engine, count in Counter(r.engine for r in records).most_common():
        typer.echo(f"{engine:12s} {count:7d}")
    typer.echo(f"\nTrained on {fitted.training_size} presets across {len(fitted.engines)} engines -> {model}")


@app.command()
def generate(
    model: ModelPath = DEFAULT_MODEL,
    engine: Annotated[str | None, typer.Option(help="Engine name or id; mixes engines when omitted")] = None,
    category: Annotated[str | None, typer.Option(help="Category, e.g. Pad, Bass, Lead")] = None,
    count: Annotated[int, typer.Option(min=1, max=BANK_SLOTS)] = 32,
    seed: Annotated[int, typer.Option()] = 1,
    bank: Annotated[str | None, typer.Option(help="Bank name [default: derived from options]")] = None,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("generated"),
) -> None:
    """Generate a bank of presets: single .mfpz files, an .mfprojz bank for MIDI Control Center, a CSV report."""
    fitted = CopulaPresetModel.load(model)
    try:
        generated = PresetGenerator(fitted).generate(
            count,
            np.random.default_rng(seed),
            engine_id=_engine_id(fitted, engine),
            category=_category_id(category),
        )
    except GenerationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    paths = export_bank(generated, out, bank or f"Generated {engine or 'Mixed'} {seed}")
    for g in generated:
        typer.echo(
            f"  {g.preset.name:14s} {g.engine:11s} {g.category_name:10s} nearest real: {g.nearest_real!r} "
            f"(distance {g.distance:.2f}, {g.distance_percentile:.0f}th percentile of real gaps)"
        )
    typer.echo(f"\nBank:    {paths.bank}\nPresets: {paths.presets_dir}\nReport:  {paths.report}")


def _engine_id(model: CopulaPresetModel, value: str | None) -> int | None:
    """Engine id from an id or a case-insensitive name; None mixes engines."""
    if value is None:
        return None
    if value.isdigit() and int(value) in model.engines:
        return int(value)
    by_name = {m.engine.lower(): eid for eid, m in model.engines.items()}
    if value.lower() not in by_name:
        names = ", ".join(m.engine for m in model.engines.values())
        raise typer.BadParameter(f"unknown engine {value!r}; choose from {names}")
    return by_name[value.lower()]


def _category_id(value: str | None) -> int | None:
    if value is None:
        return None
    by_name = {c.lower(): i for i, c in enumerate(CATEGORIES)}
    if value.lower() not in by_name:
        raise typer.BadParameter(f"unknown category {value!r}; choose from {', '.join(CATEGORIES)}")
    return by_name[value.lower()]
