import os
import onnx
from typing import Iterable
from pydantic import BaseModel, Field


class TritonTensorConfig(BaseModel):
    name: str
    dims: list[int]
    data_type: str


class TritonModelConfig(BaseModel):
    name: str
    platform: str
    input: list[TritonTensorConfig] = Field(default_factory=list)
    output: list[TritonTensorConfig] = Field(default_factory=list)

    def set_io_from_onnx(self, model: onnx.ModelProto) -> None:
        self.input, self.output = convert_onnx_to_triton_io_tensors(model)

    def save(self, model_repository_path: str) -> None:
        text = json_to_pbtxt(config=self.model_dump(), indent=2)
        with open(
            os.path.join(model_repository_path, self.name, "config.pbtxt"), "w"
        ) as f:
            f.write(text)


def convert_onnx_to_triton_tensor(tensor: onnx.ValueInfoProto) -> TritonTensorConfig:
    ONNX_TO_TRITON_DATA_TYPE_TABLE = {
        1: "TYPE_FP32",
        10: "TYPE_FP16",
        6: "TYPE_INT32",
        7: "TYPE_INT64",
        3: "TYPE_INT8",
        2: "TYPE_UINT8",
        9: "TYPE_BOOL",
        8: "TYPE_STRING",
    }  # not full

    elem_type = tensor.type.tensor_type.elem_type

    if not (elem_type in ONNX_TO_TRITON_DATA_TYPE_TABLE):
        raise ValueError(
            "Can not convert datatype, add it to ONNX_TO_TRITON_DATA_TYPE_TABLE. \
                See here https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html#datatypes"
        )

    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.HasField("dim_value") else -1)

    return TritonTensorConfig(
        name=tensor.name,
        data_type=ONNX_TO_TRITON_DATA_TYPE_TABLE[elem_type],
        dims=shape,
    )


def convert_onnx_to_triton_io_tensors(
    model: onnx.ModelProto,
) -> tuple[list[TritonTensorConfig], list[TritonTensorConfig]]:
    def _convert(tensors: Iterable) -> list[TritonTensorConfig]:
        return list(map(convert_onnx_to_triton_tensor, tensors))

    return _convert(model.graph.input), _convert(model.graph.output)


def json_to_pbtxt(config: dict, indent: int = 0) -> str:
    def _contains_config(data: list) -> bool:
        return dict in map(type, data)

    def convert_value(value) -> str:
        """Format values appropriately for config.pbtxt"""
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value)

    indent_step: str = " " * indent

    def convert_json(data: dict, prefix_indent: str) -> list[str]:
        nonlocal indent_step
        nonlocal convert_value

        lines: list[str] = []
        for key, value in data.items():
            if value is None:
                continue

            if isinstance(value, list) and _contains_config(value):
                lines.append(f"{prefix_indent}{key}: [")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{prefix_indent + indent_step}{{")
                        lines.extend(
                            convert_json(item, prefix_indent + indent_step * 2)
                        )
                        lines.append(
                            f"{prefix_indent + indent_step}}}"
                            + ("," if i + 1 < len(value) else "")
                        )
                    else:
                        raise ValueError(
                            "List contains both dict (sub-config) and some values"
                        )
                lines.append(f"{prefix_indent}]")
            elif isinstance(value, dict):
                lines.append(f"{prefix_indent}{{")
                lines.extend(convert_json(value, prefix_indent + indent_step))
                lines.append(f"{prefix_indent}}}")
            else:
                # Special case: Skip max_batch_size if 0 (Triton convention)
                if key == "max_batch_size" and value == 0:
                    continue
                lines.append(f"{prefix_indent}{key}: {convert_value(value)}")
        return lines

    return "\n".join(convert_json(data=config, prefix_indent=""))
