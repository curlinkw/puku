from puku_deployment.docker.dockerfiles import Dockerfile


class TritonDockerfile(Dockerfile):
    version: str = "25.05"
    workdir: str = "/workspace"
    dependencies: list[str] = []

    def to_text(self) -> str:
        text = ""
        text += self._from(f"nvcr.io/nvidia/tritonserver:{self.version}-py3")
        text += self._workdir(self.workdir)
        return text
