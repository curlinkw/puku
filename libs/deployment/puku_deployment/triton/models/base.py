from pydantic import BaseModel

from puku_deployment.triton.enums import InferenceBackend


class TritonModel(BaseModel):
    def export(self, path: str, backend: InferenceBackend) -> None:
        raise NotImplementedError
