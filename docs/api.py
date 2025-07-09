import os
import mkdocs_gen_files
from pathlib import Path
from itertools import chain

LIBS = ["puku", "core", "deployment"]


def write_api(lib_name: str):
    SOURCE_NAME = "puku" if lib_name == "puku" else f"puku_{lib_name}"
    SRC_PATH = Path(os.path.abspath(f"../libs/{lib_name}/{SOURCE_NAME}"))
    API_PATH = Path(f"{lib_name}/api.md")

    with mkdocs_gen_files.open(API_PATH, "w") as f:
        f.write("# API Reference\n\n")

        for path in SRC_PATH.rglob("*.py"):
            if any(
                part.startswith(".") or part == "__pycache__" or part.startswith("_")
                for part in path.parts
            ):
                continue

            module_path = path.relative_to(SRC_PATH).with_suffix("")
            module_name = ".".join(chain([SOURCE_NAME], module_path.parts))

            if not module_name:
                continue

            f.write(f"::: {module_name}\n")


for lib in LIBS:
    write_api(lib)
