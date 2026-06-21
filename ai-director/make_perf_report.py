"""
Build a performance report (PNG chart + markdown summary) from a perfmon CSV.
Usage: python make_perf_report.py <csv_path> <out_dir> [phase_log]
"""
import sys, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(out_dir, exist_ok=True)

rows = []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

if not rows:
    print("no data"); sys.exit(1)

t = [r["elapsed_s"] for r in rows]
def col(k): return [r.get(k, 0) for r in rows]
def stats(k):
    v = col(k); return (max(v), sum(v)/len(v), min(v))

gpu_total = rows[0].get("gpu_mem_total", 16303) or 16303

# ── chart: 3 stacked panels ───────────────────────────────────────────
fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
ax[0].plot(t, col("gpu_util"), color="#1f9d55", label="GPU util %")
ax[0].plot(t, col("cpu_pct"), color="#3b82f6", label="CPU util %")
ax[0].set_ylabel("Utilization %"); ax[0].set_ylim(0, 105); ax[0].legend(loc="upper right"); ax[0].grid(alpha=0.3)
ax[0].set_title("AI Director — Pipeline Performance")

ax[1].plot(t, [m/1024 for m in col("gpu_mem_mb")], color="#a855f7", label="GPU VRAM (GB)")
ax[1].axhline(gpu_total/1024, color="#ef4444", ls="--", lw=1, label=f"VRAM limit {gpu_total/1024:.1f}GB")
ax[1].plot(t, col("ram_used_gb"), color="#f59e0b", label="System RAM (GB)")
ax[1].set_ylabel("Memory (GB)"); ax[1].legend(loc="upper right"); ax[1].grid(alpha=0.3)

ax[2].plot(t, col("gpu_temp"), color="#ef4444", label="GPU temp °C")
ax[2].plot(t, col("gpu_power_w"), color="#10b981", label="GPU power (W)")
ax[2].set_ylabel("°C  /  Watts"); ax[2].set_xlabel("Elapsed (s)"); ax[2].legend(loc="upper right"); ax[2].grid(alpha=0.3)

png = os.path.join(out_dir, "performance_report.png")
plt.tight_layout(); plt.savefig(png, dpi=110); plt.close()

# ── markdown summary ──────────────────────────────────────────────────
gu = stats("gpu_util"); gt = stats("gpu_temp"); gm = stats("gpu_mem_mb")
gp = stats("gpu_power_w"); cu = stats("cpu_pct"); rg = stats("ram_used_gb"); rp = stats("ram_pct")
dur = t[-1]
md = f"""# AI Director — Performance Report

**Run duration:** {dur:.0f}s ({dur/60:.1f} min) | **Samples:** {len(rows)} (2s interval)
**GPU:** RTX 5070 Ti ({gpu_total/1024:.1f} GB)

| Metric | Peak | Avg | Min |
|--------|------|-----|-----|
| GPU utilization | {gu[0]:.0f}% | {gu[1]:.0f}% | {gu[2]:.0f}% |
| GPU temperature | {gt[0]:.0f}°C | {gt[1]:.0f}°C | {gt[2]:.0f}°C |
| GPU VRAM | {gm[0]/1024:.1f} GB | {gm[1]/1024:.1f} GB | {gm[2]/1024:.1f} GB |
| GPU power | {gp[0]:.0f} W | {gp[1]:.0f} W | {gp[2]:.0f} W |
| CPU utilization | {cu[0]:.0f}% | {cu[1]:.0f}% | {cu[2]:.0f}% |
| System RAM | {rg[0]:.1f} GB ({rp[0]:.0f}%) | {rg[1]:.1f} GB | {rg[2]:.1f} GB |

**VRAM headroom:** peak {gm[0]/1024:.1f}/{gpu_total/1024:.1f} GB — {'⚠️ near limit (spill risk)' if gm[0] > gpu_total*0.95 else 'OK, no spill'}
**Thermals:** peak {gt[0]:.0f}°C — {'⚠️ hot' if gt[0] >= 83 else 'healthy (well within limits)'}

![chart](performance_report.png)
"""
with open(os.path.join(out_dir, "performance_report.md"), "w", encoding="utf-8") as f:
    f.write(md)

print(f"Report -> {out_dir}/performance_report.md + .png")
print(f"GPU util avg {gu[1]:.0f}% peak {gu[0]:.0f}% | temp peak {gt[0]:.0f}C | VRAM peak {gm[0]/1024:.1f}GB | dur {dur/60:.1f}min")
