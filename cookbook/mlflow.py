import os
import mlflow
import mlflow.langchain as mlflow_lc

from typing import Any


def save_and_register_lc_model(
    lc_model,
    name: str,
    path: str | None = None,
    version: str | int = 1,
    tags: dict[str, Any] = {},
):
    if path is None:
        path = name

    model_path = os.path.join(f"mlruns/saved_models/{version}", path)

    mlflow_lc.save_model(lc_model=lc_model, path=model_path)

    mlflow.register_model(
        name=name, model_uri=model_path, tags={**tags, "version": version}
    )


def load_lc_model(
    model_uri: str,
    version: str | int = 1,
):
    return mlflow_lc.load_model(
        model_uri=os.path.join(f"mlruns/saved_models/{version}", model_uri)
    )
