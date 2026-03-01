# ml/feature_contract.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any
import math

SCHEMA_VERSION = "turbofan_dataset_A_v1.0.0"

FEATURES_V1 = [
    "N1_RPM",
    "N2_RPM",
    "T4",
    "m_fuel",
    "FAR",
    "m_gas",
    "P0",
    "P2",
    "P3",
    "P4",
    "P5",
    "Thrust",
    "Thrust_core",
    "Thrust_bypass",
    "Vexit_core",
    "Vexit_bypass",
    "TorqueDiff_N1",
    "TorqueDiff_N2",
    "PR_fan",
    "PR_hpc",
    "PR_core_total",
    "PR_turb",
    "TSFC",
    "ThrustSplit_bypass",
    "TorqueResSum",
    "DeltaT_turb",
]

AUX_COLUMNS = [
    "label_status",
    "label_state",
    "degradation_score",
    "fault_class",
    "schema_version",
]

# Dtypes for strict validation
DTYPES_V1: Dict[str, type] = {k: float for k in FEATURES_V1}

@dataclass(frozen=True)
class FeatureValidationResult:
    ok: bool
    errors: List[str]

def _is_finite_number(x: Any) -> bool:
    if x is None:
        return False
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)

def validate_feature_row(
    row: Dict[str, Any],
    *,
    strict: bool = True,
    require_all: bool = True,
) -> FeatureValidationResult:
    """
    Validates one feature row dict against FEATURES_V1.
    - strict=True: no extra keys allowed (except AUX columns)
    - require_all=True: all FEATURES_V1 must exist
    """
    errors: List[str] = []

    # Missing keys
    if require_all:
        for k in FEATURES_V1:
            if k not in row:
                errors.append(f"Missing key: {k}")

    # Extra keys (ignore AUX)
    allowed = set(FEATURES_V1) | set(AUX_COLUMNS)
    if strict:
        for k in row.keys():
            if k not in allowed:
                errors.append(f"Unexpected key: {k}")

    # Type + NaN/Inf checks for present feature keys
    for k in FEATURES_V1:
        if k in row:
            if not _is_finite_number(row[k]):
                errors.append(f"Non-finite value for {k}: {row[k]}")

    return FeatureValidationResult(ok=(len(errors) == 0), errors=errors)

def as_ordered_feature_vector(row: Dict[str, Any]) -> List[float]:
    """
    Convert dict -> ordered list in FEATURES_V1 order.
    Call validate_feature_row first in strict mode.
    """
    return [float(row[k]) for k in FEATURES_V1]

def export_contract_dict() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "features": FEATURES_V1,
        "aux_columns": AUX_COLUMNS,
        "dtypes": {k: "float" for k in FEATURES_V1},
    }