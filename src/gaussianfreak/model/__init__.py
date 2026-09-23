"""The Gaussian copula preset model: one copula per oscillator engine."""

from gaussianfreak.model.engine_model import EngineModel, TemplatePreset
from gaussianfreak.model.preset_model import CopulaPresetModel, ModelConfig

__all__ = ["CopulaPresetModel", "EngineModel", "ModelConfig", "TemplatePreset"]
