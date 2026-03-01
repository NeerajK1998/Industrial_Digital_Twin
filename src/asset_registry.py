from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REGISTRY_PATH = Path("data/assets.json")


@dataclass
class AssetRecord:
    asset_id: str          # unique key (we use name slug)
    name: str
    asset_type: str        # "CNC" | "Turbofan" | etc
    device_config: str     # e.g. "configs/device_template.json"
    pdm_json: str          # e.g. "data/PDM_Data.json"
    notes: str = ""


def _slug(s: str) -> str:
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in [" ", "-", "_"]:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "asset"


def _read_registry(path: Path = REGISTRY_PATH) -> Dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": "assets_v1", "assets": []}, indent=2))
    try:
        return json.loads(path.read_text())
    except Exception:
        # fallback if file is corrupted
        return {"version": "assets_v1", "assets": []}


def _write_registry(obj: Dict, path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def list_assets() -> List[AssetRecord]:
    reg = _read_registry()
    assets = reg.get("assets", [])
    out: List[AssetRecord] = []
    for a in assets:
        try:
            out.append(AssetRecord(**a))
        except Exception:
            continue
    return out


def get_asset(asset_id: str) -> Optional[AssetRecord]:
    for a in list_assets():
        if a.asset_id == asset_id:
            return a
    return None


def upsert_asset(rec: AssetRecord) -> None:
    reg = _read_registry()
    assets = reg.get("assets", [])

    # replace if exists, else append
    replaced = False
    for i, a in enumerate(assets):
        if a.get("asset_id") == rec.asset_id:
            assets[i] = asdict(rec)
            replaced = True
            break
    if not replaced:
        assets.append(asdict(rec))

    reg["assets"] = assets
    reg["version"] = reg.get("version", "assets_v1")
    _write_registry(reg)


def delete_asset(asset_id: str) -> bool:
    reg = _read_registry()
    assets = reg.get("assets", [])
    new_assets = [a for a in assets if a.get("asset_id") != asset_id]
    changed = len(new_assets) != len(assets)
    reg["assets"] = new_assets
    _write_registry(reg)
    return changed


def make_asset_id(name: str) -> str:
    return _slug(name)


def ensure_default_asset_exists() -> None:
    """Nice UX: create a default CNC asset if registry is empty."""
    assets = list_assets()
    if assets:
        return

    default = AssetRecord(
        asset_id="demo-cnc",
        name="Demo CNC",
        asset_type="CNC",
        device_config="configs/device_template.json",
        pdm_json="data/pdm/demo-cnc_pdm.json",
        notes="Auto-created default asset"
    )
    upsert_asset(default)