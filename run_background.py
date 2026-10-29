"""Run Fold 1 with durable logs from an independent Windows scheduled task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RELEASE = HERE.parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs" / "12fields_relaunch")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run_dir = args.output_dir / "fold1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "process_exit.json").unlink(missing_ok=True)
    resume_path = args.resume or (run_dir / "latest_training.pt" if (run_dir / "latest_training.pt").is_file() else None)
    started = time.time()
    command = [str(PROJECT / ".venv" / "Scripts" / "python.exe"), "-u", str(HERE / "train.py"),
               "--fold", "1", "--release-root", str(RELEASE), "--workers", "0",
               "--output-dir", str(args.output_dir)]
    if resume_path:
        command.extend(["--resume", str(resume_path)])
    code = -1
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as out, \
            (run_dir / "stderr.log").open("a", encoding="utf-8") as err:
        print(json.dumps({"event": "start", "timestamp_unix": started, "command": command}, ensure_ascii=False),
              file=out, flush=True)
        try:
            result = subprocess.run(command, cwd=HERE, stdout=out, stderr=err, check=False,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            code = result.returncode
        except BaseException:
            traceback.print_exc(file=err)
        finally:
            if code != 0:
                status_path = run_dir / "status.json"
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    status = {}
                status.update({"state": "failed", "exit_code": code, "updated_at_unix": time.time()})
                status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            (run_dir / "process_exit.json").write_text(
                json.dumps({"exit_code": code, "started_at_unix": started,
                            "ended_at_unix": time.time()}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"event": "exit", "exit_code": code}, ensure_ascii=False), file=out, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
