import os
import docker
from typing import Iterable, Optional
from docker.types import Mount, DeviceRequest
from docker.models.containers import Container

from puku_deployment.triton.docker.dockerfiles import TritonDockerfile
from puku_deployment.triton.model_repository import ModelRepository


def build_triton(
    path: str,
    dockerfile: TritonDockerfile = TritonDockerfile(),
    image_name: str = "triton",
) -> Iterable:
    client = docker.from_env()

    dockefile_path = os.path.join(path, "Dockerfile")
    dockerfile.save(path=dockefile_path)
    return client.images.build(path=path, tag=image_name)[1]


def run_triton(
    model_repository: ModelRepository,
    image_name: str = "triton",
    detach: bool = True,
    remove: bool = True,
    gpus: Optional[int] = -1,
) -> Container | bytes:
    client = docker.from_env()

    return client.containers.run(
        image=image_name,
        command=["tritonserver", "--model-repository=/workspace/models"],
        device_requests=(
            [] if gpus is None else [DeviceRequest(count=gpus, capabilities=[["gpu"]])]
        ),
        mounts=[
            Mount(
                target="/workspace/models",
                source=f"{os.path.abspath(model_repository.path)}",
                type="bind",
                read_only=True,
            )
        ],
        ports={
            "8000/tcp": 8000,  # HTTP/REST
            "8001/tcp": 8001,  # gRPC
            "8002/tcp": 8002,  # Metrics
        },
        detach=detach,
        remove=remove,
    )
