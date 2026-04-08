from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_run_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_relative(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path)


def build_local_run_manifest(
    config: dict[str, Any],
    *,
    repo_root: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    event_summary_path = _resolve_repo_relative(repo_root, config["event_summary_path"]).resolve()
    output_dir = _resolve_repo_relative(repo_root, config["output_dir"]).resolve()
    model_dir = _resolve_repo_relative(repo_root, config["model_dir"]).resolve()

    command = [
        python_executable,
        "-m",
        "option_a_tsunami_rl.training.train_ppo_lstm",
        "--event-summary-path",
        str(event_summary_path),
        "--output-dir",
        str(output_dir),
        "--model-dir",
        str(model_dir),
    ]

    for key, value in config["selected_hyperparameters"].items():
        command.extend([f"--{key}", str(value)])

    return {
        "selection_metric_name": config["selection_metric_name"],
        "selected_run_name": config["selected_run_name"],
        "working_directory": str(repo_root.resolve()),
        "event_summary_path": str(event_summary_path),
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "command": command,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build or execute a local PPO-LSTM reproducibility run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "training" / "local_repro_config.json",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=repo_root / "outputs" / "tables" / "local_repro_manifest.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_run_config(args.config)
    manifest = build_local_run_manifest(config, repo_root=repo_root)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote_manifest={args.output_manifest}")

    if args.execute:
        subprocess.run(manifest["command"], cwd=repo_root, check=True)
        print("executed=true")
    else:
        print("dry_run=true")


if __name__ == "__main__":
    main()
