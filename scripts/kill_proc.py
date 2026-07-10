import argparse
import sys
from datetime import datetime

import psutil


def report(pid: int) -> psutil.Process | None:
    try:
        p = psutil.Process(pid)
        with p.oneshot():
            info = {
                "PID": p.pid,
                "Name": p.name(),
                "Status": p.status(),
                "Started": datetime.fromtimestamp(p.create_time()).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "CMD": " ".join(p.cmdline()) or "(none)",
                "Parent PID": p.ppid(),
            }
        print("\n=== Process Report ===")
        for k, v in info.items():
            print(f"  {k:<12}: {v}")
        print()
        return p
    except psutil.NoSuchProcess:
        print(f"No process with PID {pid}.")
        return None
    except psutil.AccessDenied:
        print(f"Access denied reading process {pid}.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Report and optionally kill a process")
    parser.add_argument("pid", type=int, help="Process PID")
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Kill without prompting"
    )
    args = parser.parse_args()

    proc = report(args.pid)
    if proc is None:
        sys.exit(1)

    confirm = "y" if args.yes else input("Kill this process? [y/N]: ").strip().lower()

    if confirm == "y":
        try:
            proc.kill()
            print(f"Process {args.pid} terminated.")
        except psutil.NoSuchProcess:
            print("Process no longer exists.")
        except psutil.AccessDenied:
            print("Access denied — try running as administrator.")
    else:
        print("Aborted.")


if __name__ == "__main__":
    main()
