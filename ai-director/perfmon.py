"""
Performance monitor — samples GPU (nvidia-smi) + CPU/RAM (psutil) to a CSV.
Runs until a stop-file appears or max_seconds elapses.

Usage: python perfmon.py <csv_path> [max_seconds]
Stop:  create the file  <csv_path>.stop
"""
import sys, os, time, subprocess, csv

csv_path = sys.argv[1] if len(sys.argv) > 1 else "perf.csv"
max_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3600
stop_file = csv_path + ".stop"
if os.path.exists(stop_file):
    os.remove(stop_file)

try:
    import psutil
except Exception:
    psutil = None

GPU_Q = "timestamp,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"


def sample_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        return dict(gpu_temp=float(parts[1]), gpu_util=float(parts[2]),
                    gpu_mem_mb=float(parts[3]), gpu_mem_total=float(parts[4]),
                    gpu_power_w=float(parts[5]))
    except Exception:
        return dict(gpu_temp=0, gpu_util=0, gpu_mem_mb=0, gpu_mem_total=0, gpu_power_w=0)


t0 = time.time()
fields = ["elapsed_s", "gpu_temp", "gpu_util", "gpu_mem_mb", "gpu_mem_total",
          "gpu_power_w", "cpu_pct", "ram_pct", "ram_used_gb"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    if psutil:
        psutil.cpu_percent(interval=None)  # prime
    while True:
        elapsed = time.time() - t0
        if elapsed > max_seconds or os.path.exists(stop_file):
            break
        row = {"elapsed_s": round(elapsed, 1)}
        row.update(sample_gpu())
        if psutil:
            row["cpu_pct"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            row["ram_pct"] = vm.percent
            row["ram_used_gb"] = round(vm.used / 1e9, 2)
        else:
            row.update(cpu_pct=0, ram_pct=0, ram_used_gb=0)
        w.writerow(row)
        f.flush()
        time.sleep(2)

if os.path.exists(stop_file):
    os.remove(stop_file)
print(f"perfmon stopped after {time.time()-t0:.0f}s -> {csv_path}")
