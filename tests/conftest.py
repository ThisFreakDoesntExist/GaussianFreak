from __future__ import annotations

from pathlib import Path

import pytest

from gaussianfreak.dataset import PresetLibrary, PresetRecord
from gaussianfreak.model import CopulaPresetModel, ModelConfig
from tests.synthetic import write_training_dir


@pytest.fixture(scope="session")
def training_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_training_dir(tmp_path_factory.mktemp("training"))


@pytest.fixture(scope="session")
def records(training_dir: Path) -> list[PresetRecord]:
    return PresetLibrary(training_dir).scan().records


@pytest.fixture(scope="session")
def model(records: list[PresetRecord]) -> CopulaPresetModel:
    return CopulaPresetModel.fit(records, ModelConfig(seed=3))
