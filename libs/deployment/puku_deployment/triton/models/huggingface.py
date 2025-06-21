import os
import shutil
import subprocess
from typing import Optional

from puku_deployment.triton.models.base import TritonModel
from puku_deployment.triton.enums import InferenceBackend


class OptimumCLIException(Exception):
    pass


class HuggingFaceModel(TritonModel):
    huggingface_model_name: str
    task: Optional[str] = None
    optimize: Optional[str] = None
    precision: Optional[str] = None

    def export(self, path: str, backend: InferenceBackend) -> None:
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists")

        export_response = HuggingFaceModel._optimum_cli_export(
            self.huggingface_model_name,
            path,
            task=self.task,
            optimize=self.optimize,
        )
        if export_response.returncode:
            raise OptimumCLIException(export_response.stderr)
        else:
            print(export_response.stdout)

        if not (self.precision is None):
            quantized_path = path + "_quantized"

            if os.path.exists(quantized_path):
                raise FileExistsError(f"{quantized_path} exists")

            quantize_response = HuggingFaceModel._optimum_cli_quantize(
                model_path=path,
                output_path=quantized_path,
                precision=self.precision,
            )

            if quantize_response.returncode:
                raise OptimumCLIException(quantize_response.stderr)
            else:
                print(quantize_response.stdout)
                shutil.rmtree(path)
                os.rename(quantized_path, path)

    @staticmethod
    def _optimum_cli_export(
        model_name: str,
        path: str,
        task: Optional[str] = None,
        optimize: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        cmd = [
            "optimum-cli",
            "export",
            "onnx",
        ]

        if not (task is None):
            cmd += ["--task", task]

        if not (optimize is None):
            cmd += ["--optimize", optimize]

        cmd += [
            "--model",
            model_name,
            path,
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    @staticmethod
    def _optimum_cli_quantize(
        model_path: str, output_path: str, precision: str
    ) -> subprocess.CompletedProcess:
        cmd = (
            ["optimum-cli", "onnxruntime", "quantize", "--onnx_model", model_path]
            + ["--" + precision]
            + ["-o", output_path]
        )
        return subprocess.run(cmd, capture_output=True, text=True)
