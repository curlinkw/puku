from pydantic import BaseModel


class Dockerfile(BaseModel):
    def _from(self, image: str) -> str:
        return f"""FROM {image}\n\n"""

    def _workdir(self, path: str) -> str:
        return f"""WORKDIR {path}\n\n"""

    def _run(self, commands: list[str]) -> str:
        if not commands:
            return ""

        text = "RUN " + "\\\n".join(commands)
        return text

    def add_dependencies(self, dependencies: list[str]) -> str:
        if not dependencies:
            return ""

        return self._run(
            ["apt-get update &&", "apt-get install -y --no-install-recommends"]
            + dependencies
            + ["&& rm -rf /var/lib/apt/lists/*"]
        )

    def to_text(self) -> str:
        raise NotImplementedError
