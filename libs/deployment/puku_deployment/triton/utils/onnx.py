import onnx
import numpy as np
from typing import Tuple, Optional, Iterable
from onnx import helper, shape_inference
from spox import Tensor, Var, argument, inline, build
from itertools import chain

from puku_core._api import experimental


@experimental()
def add_batch_dimension(
    model: onnx.ModelProto,
    batch_inputs: Optional[list[str]] = None,
    batch_dim="batch_size",
) -> onnx.ModelProto:
    """Adds batch dimension to an ONNX model while preserving other dimensions."""

    for input_tensor in chain(model.graph.input, model.graph.output):
        if not ((batch_inputs is None) or (input_tensor.name in batch_inputs)):
            continue
        # Get the input tensor and its original dimensions
        original_dims = input_tensor.type.tensor_type.shape.dim  # This was missing!

        # Create new shape object
        new_shape = input_tensor.type.tensor_type.shape

        # Clear existing dimensions
        new_shape.ClearField("dim")

        # 1. Add batch dimension first
        batch = new_shape.dim.add()
        if isinstance(batch_dim, int):
            batch.dim_value = batch_dim  # Fixed batch size
        else:
            batch.dim_param = batch_dim  # Dynamic batch size ('batch_size' or 'N')

        # 2. Copy remaining dimensions from original
        for orig_dim in original_dims:
            new_dim = new_shape.dim.add()

            # Handle defined dimensions
            if orig_dim.HasField("dim_param"):
                new_dim.dim_param = orig_dim.dim_param
            elif orig_dim.HasField("dim_value"):
                new_dim.dim_value = orig_dim.dim_value
            else:
                # Handle undefined dimensions
                new_dim.dim_param = f"dim_{len(new_shape.dim)-1}"  # Default naming

    # Run shape inference
    return shape_inference.infer_shapes(model)


def convert_onnx_to_spox_tensor(tensor: onnx.ValueInfoProto) -> Tuple[str, Tensor]:
    dtype = helper.tensor_dtype_to_np_dtype(tensor.type.tensor_type.elem_type)

    if dtype == np.dtype("object"):
        dtype = np.dtype("str")

    dims = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)  # Static dimension
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)  # Dynamic dimension (symbolic name)
        else:
            dims.append(None)  # Unknown dimension

    return tensor.name, Tensor(dtype=dtype, shape=dims)  # type: ignore


def convert_onnx_arguments_to_spox(
    arguments: Iterable[onnx.ValueInfoProto],
) -> dict[str, Var]:
    return {
        name: argument(tensor)
        for name, tensor in map(convert_onnx_to_spox_tensor, arguments)
    }


def sequential(
    first_model: onnx.ModelProto,
    second_model: onnx.ModelProto,
    io_map: Optional[dict[str, str]] = None,
) -> onnx.ModelProto:
    """Merger two onnx models. Discussed in:
    https://github.com/onnx/onnx/issues/5006#issuecomment-1500402402
    """
    inputs = convert_onnx_arguments_to_spox(first_model.graph.input)

    if not io_map:
        io_map = {
            name: name
            for name in convert_onnx_arguments_to_spox(second_model.graph.input).keys()
        }

    intermediate = inline(first_model)(**inputs)

    intermediate = {
        input_name: intermediate[output_name]
        for input_name, output_name in io_map.items()
    }

    outputs = inline(second_model)(**intermediate)
    combined = build(inputs=inputs, outputs=outputs)
    return combined
