"""Monitor migration progress from a separate terminal.

Run in a new terminal while migrate_bronze_prices.py is running:
    uv run scripts/monitor_migration.py
"""

import os
import time
from datetime import datetime
from pathlib import Path

import psutil

DB_PATH = Path("data/bronze/cards.duckdb")
WAL_PATH = Path("data/bronze/cards.duckdb.wal")
INTERVAL = 30  # seconds


def find_migration_process():
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_times", "memory_info"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "migrate_bronze_prices" in cmdline:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def fmt_mb(n_bytes):
    return f"{n_bytes / 1024 / 1024:,.0f} MB"


def fmt_delta(delta_bytes):
    if delta_bytes == 0:
        return "  —  "
    sign = "+" if delta_bytes > 0 else ""
    return f"{sign}{delta_bytes / 1024 / 1024:,.1f} MB"


print(f"{'CZAS':>15} | {'DB ROZMIAR':>12} | {'DELTA DB':>10} | {'WAL':>8} | {'CPU':>7} | {'RAM':>6}")
print("-" * 80)

prev_db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
start_time = time.time()

while True:
    now = datetime.now().strftime("%H:%M:%S")
    elapsed = int(time.time() - start_time)
    elapsed_str = f"{elapsed // 3600}h{(elapsed % 3600) // 60:02d}m"

    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    wal_size = WAL_PATH.stat().st_size if WAL_PATH.exists() else 0
    delta = db_size - prev_db_size

    proc = find_migration_process()
    if proc:
        try:
            cpu_s = int(proc.cpu_times().user + proc.cpu_times().system)
            ram_gb = proc.memory_info().rss / 1024 / 1024 / 1024
            proc_info = f"{cpu_s:>6}s | {ram_gb:>4.1f} GB"
            alive = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_info = "  koniec?  "
            alive = False
    else:
        proc_info = "  BRAK  |  —  "
        alive = False

    print(
        f"{now} +{elapsed_str} | {fmt_mb(db_size):>12} | {fmt_delta(delta):>10} | "
        f"{fmt_mb(wal_size):>8} | {proc_info}"
    )

    if not alive and not WAL_PATH.exists():
        print()
        print("Proces zakończony i WAL zniknął — migracja gotowa!")
        break

    prev_db_size = db_size
    time.sleep(INTERVAL)
