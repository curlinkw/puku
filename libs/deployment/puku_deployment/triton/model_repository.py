from typing import Optional
from pydantic import BaseModel, Field

from puku_deployment.triton.models.base import TritonModel, TritonEnsembleModel
from puku_deployment.triton.models.config import TritonModelConfig


class ModelRepository(BaseModel):
    path: str
    models: dict[str, TritonModel] = Field(default_factory=dict)
    configs: dict[str, TritonModelConfig] = Field(default_factory=dict)

    def add_model(self, model: TritonModel):
        config = model.save(self.path)
        self.models[config.name] = model
        self.configs[config.name] = config

    def add_models(self, models: list[TritonModel]):
        for model in models:
            self.add_model(model)

    def add_ensemble(
        self,
        name: str,
        models: list[str],
        io_maps: Optional[list[tuple[dict[str, str], dict[str, str]]]] = None,
    ):
        configs = [self.configs[model_name] for model_name in models]
        model: TritonEnsembleModel = TritonEnsembleModel.from_configs(
            name=name, configs=configs, io_maps=io_maps
        )
        self.add_model(model)
