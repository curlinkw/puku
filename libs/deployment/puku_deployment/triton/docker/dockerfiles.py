import os
from puku_deployment.docker.dockerfiles import Dockerfile


class TritonDockerfile(Dockerfile):
    version: str = "24.12"
    workdir: str = "/workspace"
    dependencies: list[str] = []

    def to_text(self) -> str:
        text = ""
        text += self._from(f"nvcr.io/nvidia/tritonserver:{self.version}-py3")
        text += self._workdir(self.workdir)
        return text

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_text())
