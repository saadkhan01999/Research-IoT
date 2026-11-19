"""Record the machine and library versions, so every result can name its hardware."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

LIBRARIES = (
    "numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "lightgbm", "imbalanced-learn",
    "scipy", "matplotlib", "PyYAML", "tqdm", "psutil", "joblib", "torch", "shap",
    "pytest", "ruff", "mypy",
)
TOOLS = ("git", "node", "npm")
GB = 1024**3


def cpu_name() -> str:
    """Readable CPU model name (Windows registry first, then the platform module)."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def library_versions(names: tuple[str, ...] = LIBRARIES) -> dict[str, str | None]:
    """Installed version of each library, or None when it is not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def tool_versions(names: tuple[str, ...] = TOOLS) -> dict[str, str | None]:
    """First line of ``<tool> --version`` for command-line tools, or None if missing."""
    versions: dict[str, str | None] = {}
    for name in names:
        exe = shutil.which(name)
        if exe is None:
            versions[name] = None
            continue
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
        text = (out.stdout or out.stderr).strip()
        versions[name] = text.splitlines()[0] if text else None
    return versions


def collect(project_root: Path) -> dict[str, Any]:
    """Machine, OS, Python, disk, GPU and library facts as one JSON-ready dict."""
    disk = shutil.disk_usage(project_root)
    return {
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "cpu": cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / GB, 1),
        "ram_available_gb": round(psutil.virtual_memory().available / GB, 1),
        "disk_free_gb_project_drive": round(disk.free / GB, 1),
        "nvidia_gpu_driver": shutil.which("nvidia-smi") is not None,
        "libraries": library_versions(),
        "tools": tool_versions(),
    }
