from enum import Enum


class InferenceBackend(Enum):
    onnx = "onnx"
    openvino = "openvino"
    

