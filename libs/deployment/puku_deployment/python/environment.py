import os
import shutil
import tempfile
import subprocess


def create_packed_conda_environment(
    path: str,
    dependencies: list[str],
    python_version: str = "3.12",
    clean_cache: bool = True,
):
    """
    Creates a temporary conda environment with complete isolation,
    installs dependencies, packs it, and cleans up without any traces.

    Ensures:
    - PYTHONNOUSERSITE=True prevents user-site packages
    - Isolated package cache in temp directory
    - Full cleanup of all temporary artifacts
    - Optional global cache cleaning

    Args:
        path: Output path for the .tar.gz environment
        dependencies: List of packages (e.g., ["numpy", "pandas"])
        python_version: Python version string
        clean_cache: Clean global conda cache after operation
    """
    # Create isolated environment configuration
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"  # Prevent user-site package interference

    with tempfile.TemporaryDirectory() as tmpdir:
        env_path = os.path.join(tmpdir, "temp_env")
        # Create isolated package cache
        env["CONDA_PKGS_DIRS"] = pkg_cache = os.path.join(tmpdir, "conda_pkgs")
        os.makedirs(pkg_cache, exist_ok=True)

        # Step 1: Create environment with isolated settings
        create_cmd = [
            "conda",
            "create",
            "--yes",
            "--quiet",
            "--prefix",
            env_path,
            "--channel",
            "conda-forge",  # Add conda-forge as primary channel
            f"python={python_version}",
        ] + dependencies

        subprocess.run(create_cmd, check=True, env=env)

        # Step 2: Pack environment using same isolation
        pack_cmd = ["conda-pack", "--quiet", "--prefix", env_path, "--output", path]
        subprocess.run(pack_cmd, check=True, env=env)

        # Step 3: Verify environment removal
        shutil.rmtree(env_path, ignore_errors=True)

    # Step 5: Optional global cache cleaning
    if clean_cache:
        clean_cmd = ["conda", "clean", "--all", "--yes", "--quiet"]
        subprocess.run(clean_cmd, check=True)
