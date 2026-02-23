from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List


def append_timeseries_row(path: str | Path, row: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
