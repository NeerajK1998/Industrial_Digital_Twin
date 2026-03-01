from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

PDM_DIR = Path("data/pdm")


def pdm_path_for_asset(asset_id: str) -> Path:
    PDM_DIR.mkdir(parents=True, exist_ok=True)
    return PDM_DIR / f"{asset_id}_pdm.json"


def load_or_create_pdm(asset_id: str) -> Dict:
    path = pdm_path_for_asset(asset_id)
    if not path.exists():
        # start EMPTY (your requirement)
        path.write_text(json.dumps({}, indent=2))
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_pdm(asset_id: str, pdm: Dict) -> None:
    path = pdm_path_for_asset(asset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pdm, indent=2))