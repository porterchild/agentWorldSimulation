import argparse
import os
import sys
from pathlib import Path
import config
import llm
from world import World


class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def claim_run_slot() -> tuple[int, Path, Path]:
    """Atomically claim the next available run number. Returns (n, run_log, full_log)."""
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    n = 1
    while True:
        run_log = logs_dir / f"run{n}.log"
        try:
            # O_EXCL ensures only one process can create the file
            fd = os.open(run_log, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return n, run_log, logs_dir / f"run{n}Full.log"
        except FileExistsError:
            n += 1


def main():
    parser = argparse.ArgumentParser(description="Run a world simulation.")
    parser.add_argument("scenario", help="The scenario to simulate")
    parser.add_argument("--turns", type=int, default=5, help="Number of turns to run (default: 5)")
    parser.add_argument("--thinking", action="store_true", help="Enable reasoning scaffold in prompts")
    parser.add_argument("--budget", type=int, default=None, help="Spawn budget (default: from config)")
    parser.add_argument("--no-log", action="store_true", help="Disable log file output")
    args = parser.parse_args()

    if args.thinking:
        config.THINKING = True
    if args.budget is not None:
        config.SPAWN_BUDGET = args.budget

    tee = None
    if not args.no_log:
        n, run_log, full_log = claim_run_slot()

        tee = Tee(run_log)
        sys.stdout = tee
        llm.set_full_log(full_log)
        print(f"Logging to {run_log}")
        print(f"Full prompt log: {full_log}\n", flush=True)

    try:
        world = World(scenario=args.scenario, spawn_budget=config.SPAWN_BUDGET)
        world.run(turns=args.turns)
    finally:
        if tee:
            sys.stdout = tee.terminal
            tee.close()
            print(f"\nLogs saved: {run_log} | {full_log}")


if __name__ == "__main__":
    main()
