import os
import onnx
from typing import Optional, Self, Optional
from pydantic import BaseModel, Field

from puku_deployment.triton.models.config import (
    TritonModelConfig,
    TritonONNXModelConfig,
    TritonPythonModelConfig,
    TritonEnsembleModelConfig,
    TritonTensorConfig,
    TritonEnsembleStepConfig,
    DataMapType,
    DataMaps,
    StepMaps,
    StepMapType,
    Parameters,
)
from puku_deployment.python.environment import create_packed_conda_environment
from puku_deployment.utils.os import create_path


class TritonModel(BaseModel):
    def get_config(self, **kwargs) -> TritonModelConfig:
        raise NotImplementedError

    def save(self, model_repository_path: str, **kwargs) -> TritonModelConfig:
        raise NotImplementedError


class TritonONNXModel(TritonModel):
    save_cache: bool = True
    _model_onnx_cache: Optional[onnx.ModelProto] = None

    def export_onnx(self, path: str) -> None:
        raise NotImplementedError

    def get_config(self, **kwargs) -> TritonONNXModelConfig:
        raise NotImplementedError

    def save(self, model_repository_path: str, **kwargs) -> TritonONNXModelConfig:
        config: TritonONNXModelConfig = self.get_config(**kwargs)
        model_path = os.path.join(model_repository_path, config.name, "1", "model.onnx")
        create_path(model_path)
        self.export_onnx(path=model_path)

        # Explicit set of io from onnx
        model_onnx: onnx.ModelProto = (
            onnx.load(model_path)
            if (self._model_onnx_cache is None)
            else self._model_onnx_cache
        )
        config.set_io_from_onnx(model_onnx)

        config.save(model_repository_path=model_repository_path)
        return config


class TritonPythonModel(TritonModel):
    python_version: str = "3.12"
    dependencies: list[str] = Field(default_factory=list)

    def get_model_code(self) -> str:
        """Get model.py code in the form of
        https://github.com/triton-inference-server/python_backend#usage
        """
        raise NotImplementedError

    def export_python(self, path: str) -> None:
        code = self.get_model_code()
        with open(path, "w") as f:
            f.write(code)

    def export_conda_environment(self, path: str):
        create_packed_conda_environment(
            path=path,
            dependencies=self.dependencies,
            python_version=self.python_version,
        )

    def get_config(self, **kwargs) -> TritonPythonModelConfig:
        raise NotImplementedError

    def save(self, model_repository_path: str, **kwargs) -> TritonPythonModelConfig:
        conda_environment_name = f"python{self.python_version}.tar.gz"

        # Configuration
        config = self.get_config(**kwargs)
        config.parameters = Parameters(
            key="EXECUTION_ENV_PATH",
            value=f"$$TRITON_MODEL_DIRECTORY/{conda_environment_name}",
        )

        # Set up paths
        model_dir = os.path.join(model_repository_path, config.name)
        model_path = os.path.join(model_dir, "1", "model.py")
        conda_environment_path = os.path.join(model_dir, conda_environment_name)

        # Export
        create_path(model_path)
        self.export_conda_environment(path=conda_environment_path)
        self.export_python(path=model_path)
        config.save(model_repository_path=model_repository_path)

        return config


class TritonEnsembleModel(TritonModel):
    """Triton ensemble model, see here:
    https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html#ensemble-models
    """

    config: TritonEnsembleModelConfig

    @classmethod
    def from_configs(
        cls,
        name: str,
        configs: list[TritonModelConfig],
        io_maps: Optional[list[tuple[dict[str, str], dict[str, str]]]] = None,
    ) -> Self:
        if io_maps is None:
            io_maps = []
            for config in configs:
                input_maps = dict()
                output_maps = dict()

                for tensor_config in config.input:
                    input_maps[tensor_config.name] = tensor_config.name

                for tensor_config in config.output:
                    output_maps[tensor_config.name] = tensor_config.name

                io_maps.append((input_maps, output_maps))

        input_flow = set()
        output_flow = set()

        possible_inputs: dict[str, TritonTensorConfig] = dict()
        possible_outputs: dict[str, TritonTensorConfig] = dict()

        for io_map, config in zip(io_maps, configs):
            input_map, output_map = io_map

            for internal, external in input_map.items():
                for _input in config.input:
                    if _input.name == internal:
                        possible_inputs[external] = _input
                        break

            for internal, external in output_map.items():
                for _output in config.output:
                    if _output.name == internal:
                        possible_outputs[external] = _output
                        break

            input_flow.update(output_map.values())
            output_flow.update(input_map.values())

        inputs: list[TritonTensorConfig] = []
        outputs: list[TritonTensorConfig] = []

        for possible_input, tensor_config in possible_inputs.items():
            if not (possible_input in input_flow):
                inputs.append(tensor_config)

        for possible_output, tensor_config in possible_outputs.items():
            if not (possible_output in output_flow):
                outputs.append(tensor_config)

        steps: list[TritonEnsembleStepConfig] = []
        for io_map, config in zip(io_maps, configs):
            input_map, output_map = io_map

            input_maps = [
                DataMapType(key=key, value=value) for key, value in input_map.items()
            ]
            output_maps = [
                DataMapType(key=key, value=value) for key, value in output_map.items()
            ]

            steps.append(
                TritonEnsembleStepConfig(
                    model_name=config.name,
                    input_map=DataMaps(maps=input_maps),
                    output_map=DataMaps(maps=output_maps),
                )
            )

        return cls(
            config=TritonEnsembleModelConfig(
                name=name,
                input=inputs,
                output=outputs,
                ensemble_scheduling=StepMaps(maps=[StepMapType(step=steps)]),
            )
        )

    def get_config(self, **kwargs) -> TritonEnsembleModelConfig:
        return self.config

    def save(self, model_repository_path: str, **kwargs) -> TritonEnsembleModelConfig:
        config = self.config
        config.save(model_repository_path=model_repository_path)
        return config
