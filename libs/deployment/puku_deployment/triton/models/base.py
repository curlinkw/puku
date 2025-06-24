import os
import onnx
from typing import Optional
from pydantic import BaseModel

from puku_deployment.triton.models.config import TritonModelConfig


class TritonModel(BaseModel):
    def get_config(self, **kwargs) -> TritonModelConfig:
        raise NotImplementedError

    def save(self, model_repository_path: str, **kwargs) -> None:
        raise NotImplementedError


class TritonONNXModel(TritonModel):
    save_cache: bool = True
    _model_onnx_cache: Optional[onnx.ModelProto] = None

    def export_onnx(self, path: str) -> None:
        raise NotImplementedError

    def save(self, model_repository_path: str, **kwargs) -> None:
        config: TritonModelConfig = self.get_config(**kwargs)
        model_path = os.path.join(model_repository_path, config.name, "model.onnx")
        self.export_onnx(path=model_path)
        model_onnx: onnx.ModelProto = (
            onnx.load(model_path)
            if (self._model_onnx_cache is None)
            else self._model_onnx_cache
        )
        config.set_io_from_onnx(model_onnx)
        config.save(model_repository_path=model_repository_path)


class TritonPythonModel(TritonModel):
    def get_model_code(self) -> str:
        """Get model.py code in the form of
        https://github.com/triton-inference-server/python_backend#usage
        """
        raise NotImplementedError

    def export_python(self, path: str) -> None:
        code = self.get_model_code()
        with open(path, "w") as f:
            f.write(code)


class TritonEnsembleModel(TritonModel):
    pass
