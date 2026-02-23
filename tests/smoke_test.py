from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]  # python_port/
sys.path.insert(0, str(ROOT))


def _pick_first_file(folder: Path, suffix: str) -> Path:
    files = sorted([p for p in folder.glob(f"*{suffix}") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {folder}")
    return files[0]


def main() -> int:
    print("=== SMOKE TEST: dtwin python_port ===")
    print("Root:", ROOT)

    cfg_dir = ROOT / "configs"
    data_dir = ROOT / "data"

    device_cfg = _pick_first_file(cfg_dir, ".json")
    pdm_json = _pick_first_file(data_dir, ".json")

    print("Using device config:", device_cfg.name)
    print("Using PDM json     :", pdm_json.name)

    main_py = ROOT / "src" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Expected entrypoint not found: {main_py}")

    print("\nRunning: python src/main.py")
    result = subprocess.run(
        [sys.executable, str(main_py)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("\n--- stdout ---\n", result.stdout)
        print("\n--- stderr ---\n", result.stderr)
        raise RuntimeError(f"src/main.py failed with code {result.returncode}")

    out_dir = ROOT / "outputs"
    runsummary = out_dir / "RunSummary.csv"
    live_ts = out_dir / "live_timeseries.csv"

    if not runsummary.exists():
        raise FileNotFoundError(f"Missing expected output: {runsummary}")
    if not live_ts.exists():
        raise FileNotFoundError(f"Missing expected output: {live_ts}")

    print("\nOutputs OK:")
    print(" -", runsummary)
    print(" -", live_ts)

    print("\n✅ SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
