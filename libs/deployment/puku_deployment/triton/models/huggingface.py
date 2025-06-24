import os
import onnx
import shutil
import subprocess
from typing import Optional
from transformers import AutoTokenizer
from onnxruntime_extensions import gen_processing_models

from puku_core._api import experimental
from puku_deployment.triton.models.base import (
    TritonONNXModel,
    TritonPythonModel,
    TritonEnsembleModel,
)
from puku_deployment.triton.models.config import TritonModelConfig


class OptimumCLIException(Exception):
    pass


class HuggingFaceONNXModel(TritonONNXModel):
    huggingface_model_name: str
    task: Optional[str] = None
    optimize: Optional[str] = None
    precision: Optional[str] = None

    def export_onnx(self, path: str) -> None:
        _SUFFIX = ".onnx"
        _DEFAULT_OPTIMUM_MODEL_NAME = "model.onnx"
        _DEFAULT_OPTIMUM_QUANTIZED_MODEL_NAME = "model_quantized.onnx"

        if path.endswith(_SUFFIX):
            path = path[: -1 * len(_SUFFIX)]

        model_path = path + _SUFFIX

        if os.path.exists(path) or os.path.exists(model_path):
            raise FileExistsError(f"{path} exists")

        if not (self._model_onnx_cache is None):
            onnx.save(self._model_onnx_cache, model_path)
            return None

        export_response = HuggingFaceONNXModel._optimum_cli_export(
            self.huggingface_model_name,
            path,
            task=self.task,
            optimize=self.optimize,
        )
        if export_response.returncode:
            raise OptimumCLIException(export_response.stderr)
        else:
            print(export_response.stdout)

        def _clear_output():
            if not os.path.exists(model_path):
                shutil.copy2(
                    os.path.join(path, f"{_DEFAULT_OPTIMUM_MODEL_NAME}"), model_path
                )
            shutil.rmtree(path)

        if not (self.precision is None):
            quantized_path = path + "_quantized"

            if os.path.exists(quantized_path):
                raise FileExistsError(f"{quantized_path} exists")

            quantize_response = HuggingFaceONNXModel._optimum_cli_quantize(
                model_path=path,
                output_path=quantized_path,
                precision=self.precision,
            )

            if quantize_response.returncode:
                _clear_output()
                raise OptimumCLIException(quantize_response.stderr)
            else:
                print(quantize_response.stdout)
                shutil.copy2(
                    os.path.join(quantized_path, _DEFAULT_OPTIMUM_QUANTIZED_MODEL_NAME),
                    model_path,
                )
                shutil.rmtree(quantized_path)

        _clear_output()

        if self.save_cache:
            self._model_onnx_cache = onnx.load(model_path)

    def get_config(self, **kwargs) -> TritonModelConfig:
        return TritonModelConfig(
            name=self.huggingface_model_name.replace("/", "_"),
            platform="onnxruntime_onnx",
        )

    @experimental(
        addendum="""Insufficient support for a onnx tokenizers by HF and Triton."""
    )
    def export_tokenizer_onnx(self, path: str):
        tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name)
        preprocess_model = gen_processing_models(
            tokenizer, pre_kwargs={"padding": True, "truncate": True}
        )[0]
        # preprocess_model = add_batch_dimension(preprocess_model, batch_inputs=None)  # type: ignore
        onnx.save(preprocess_model, path)  # type: ignore

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

        cmd += ["--model", model_name, path]
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


class HFTokenizerPythonModel(TritonPythonModel):
    huggingface_model_name: str
    input_name: str = "text"
    output_input_ids_name: str = "input_ids"
    output_attention_mask_name: str = "attention_mask"

    def get_model_code(self) -> str:
        return f"""
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    def initialize(self, args):
        \"\"\"
        Initialize the tokenizer during model loading.
        \"\"\"
        self.model_config = json.loads(args["model_config"])

        try:
            # Load tokenizer from model repository
            self.tokenizer = AutoTokenizer.from_pretrained("{self.huggingface_model_name}")
            logger.info("Tokenizer loaded successfully")

            # Validate output configuration
            output_config = pb_utils.get_output_config_by_name(self.model_config, "{self.output_input_ids_name}")
            self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        except Exception as e:
            logger.error("Initialization failed: " + str(e))
            raise

    def execute(self, requests):
        \"\"\"
        Process requests and return tokenized outputs.
        \"\"\"
        responses = []

        for request in requests:
            try:
                # Get input tensor (byte strings)
                input_text = pb_utils.get_input_tensor_by_name(request, "{self.input_name}")
                text_batch = [t.decode("UTF-8") for t in input_text.as_numpy().flatten()]

                # Tokenize batch
                tokenized = self.tokenizer(
                    text_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="np",
                    max_length=self.tokenizer.model_max_length
                )

                # Create output tensors
                input_ids = tokenized["input_ids"].astype(self.output_dtype)
                attention_mask = tokenized["attention_mask"].astype(self.output_dtype)

                # Build Triton response
                out_tensors = [
                    pb_utils.Tensor("{self.output_input_ids_name}", input_ids),
                    pb_utils.Tensor("{self.output_attention_mask_name}", attention_mask)
                ]
                responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))

            except Exception as e:
                logger.error("Request processing failed: " + str(e))
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Processing error: " + str(e))
                ))

        return responses
"""


class HFTransformerModel(TritonEnsembleModel):
    pass
