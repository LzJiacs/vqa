from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path


QUERY = (
    "timestamp,name,index,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory"
)


def sample() -> list[list[str]]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={QUERY}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")

    rows: list[list[str]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([x.strip() for x in line.split(",")])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="CSV output path")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--duration", type=float, default=0.0, help="Total monitor duration in seconds; 0 means forever")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "wall_time",
        "gpu_timestamp",
        "name",
        "index",
        "utilization_gpu_pct",
        "utilization_mem_pct",
        "memory_used_mib",
        "memory_total_mib",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "sm_clock_mhz",
        "mem_clock_mhz",
    ]

    start = time.time()
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        while True:
            now = datetime.now().isoformat(timespec="seconds")
            for row in sample():
                writer.writerow([now] + row)
            f.flush()

            if args.duration > 0 and (time.time() - start) >= args.duration:
                break
            time.sleep(max(args.interval, 0.2))


if __name__ == "__main__":
    main()
