from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

TMP_ROOT = Path.cwd() / "data" / "processed" / "pytest_tmp"


@pytest.fixture()
def tmp_path() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
