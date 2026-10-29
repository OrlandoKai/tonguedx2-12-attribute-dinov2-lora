"""Wait for the Fold 1 training task, then export detailed held-out metrics."""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUN = PROJECT / "runs" / "12fields_relaunch" / "fold1"
WATCH_STATUS = RUN / "evaluation_status.json"


def write_status(state: str, **details: object) -> None:
    WATCH_STATUS.write_text(json.dumps({"state": state, "updated_at_unix": time.time(), **details},
                                       ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    deadline = time.time() + 24 * 3600
    write_status("waiting_for_training")
    while time.time() < deadline:
        try:
            training = json.loads((RUN / "status.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            training = {}
        if training.get("state") == "complete" and (RUN / "metrics.json").is_file():
            break
        if training.get("state") == "failed":
            write_status("training_failed", training_exit_code=training.get("exit_code"))
            return 1
        exit_file = RUN / "process_exit.json"
        if exit_file.is_file():
            try:
                exit_data = json.loads(exit_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                exit_data = {}
            if exit_data.get("exit_code") not in (None, 0):
                write_status("training_failed", training_exit_code=exit_data["exit_code"])
                return 1
        time.sleep(60)
    else:
        write_status("timed_out", hours=24)
        return 2

    write_status("evaluating")
    command = [str(PROJECT / ".venv" / "Scripts" / "python.exe"), "-u",
               str(HERE / "evaluate_detailed.py"), "--run-dir", str(RUN)]
    with (RUN / "evaluation_stdout.log").open("w", encoding="utf-8") as out, \
            (RUN / "evaluation_stderr.log").open("w", encoding="utf-8") as err:
        try:
            completed = subprocess.run(command, cwd=HERE, stdout=out, stderr=err,
                                       check=False, creationflags=subprocess.CREATE_NO_WINDOW)
            code = completed.returncode
        except BaseException:
            traceback.print_exc(file=err)
            code = -1
    expected = [RUN / "detailed_metrics.json", RUN / "per_attribute_metrics.csv",
                RUN / "per_class_metrics.csv", RUN / "report.md"]
    if code == 0 and all(path.is_file() and path.stat().st_size > 0 for path in expected):
        write_status("complete", outputs=[str(path) for path in expected])
        return 0
    write_status("evaluation_failed", exit_code=code)
    return code if code else 3


if __name__ == "__main__":
    raise SystemExit(main())
