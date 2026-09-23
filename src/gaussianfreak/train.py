"""Scan the training directory, fit the copula model and save it."""

from __future__ import annotations

import logging
from pathlib import Path

from gaussianfreak.dataset import LibraryScan, PresetLibrary
from gaussianfreak.model import CopulaPresetModel, ModelConfig

logger = logging.getLogger(__name__)


def train(training_dir: Path, model_path: Path, seed: int = 0) -> tuple[CopulaPresetModel, LibraryScan]:
    scan = PresetLibrary(training_dir).scan()
    records = scan.records
    if not records:
        raise RuntimeError(f"no training-ready presets under {training_dir}; see training/README.md")
    logger.info("fitting copula model on %d presets (seed %d)", len(records), seed)
    model = CopulaPresetModel.fit(records, ModelConfig(seed=seed))
    model.save(model_path)
    logger.info("saved model to %s", model_path)
    return model, scan
