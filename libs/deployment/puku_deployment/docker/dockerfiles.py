from typing import Optional, List
from pydantic import BaseModel, Field


class Dockerfile(BaseModel):
    system_dependencies: List[str] = Field(default_factory=list)
    pip_dependencies: List[str] = Field(default_factory=list)

    pip_index_url: Optional[str] = None  # For custom package indexes
    pip_trusted_host: Optional[str] = None  # For trusted hosts

    def _from(self, image: str) -> str:
        return f"""FROM {image}\n\n"""

    def _workdir(self, path: str) -> str:
        return f"""WORKDIR {path}\n\n"""

    def _run(self, commands: list[str]) -> str:
        if not commands:
            return ""

        text = "RUN " + "\\\n".join(commands)
        return text

    def add_system_dependencies(self, system_dependencies: list[str] = []) -> str:
        system_dependencies = list(set(self.system_dependencies + system_dependencies))

        if not system_dependencies:
            return ""

        commands = [
            "apt-get update -y",
            "apt-get install -y --no-install-recommends "
            + " ".join(system_dependencies),
            "rm -rf /var/lib/apt/lists/*",
        ]
        return self._run(commands)

    def add_pip_dependencies(self, pip_dependencies: list[str] = []) -> str:
        pip_dependencies = list(set(self.pip_dependencies + pip_dependencies))

        if not pip_dependencies:
            return ""

        pip_options = []
        if not (self.pip_index_url is None):
            pip_options.append(f"--index-url {self.pip_index_url}")
        if not (self.pip_trusted_host is None):
            pip_options.append(f"--trusted-host {self.pip_trusted_host}")

        pip_command = "pip install --no-cache-dir"
        if pip_options:
            pip_command += " " + " ".join(pip_options)

        # Install dependencies in single RUN to optimize layer caching
        return self._run([pip_command, "    " + " \\\n    ".join(pip_dependencies)])

    def to_text(self) -> str:
        raise NotImplementedError
