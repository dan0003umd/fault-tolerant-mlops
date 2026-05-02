# fix_all.py — run this from D:\fault-tolerant-mlops
import os

# ── 1. generate_synthetic_log.py ─────────────────────────────
gen = """import random, csv, os, math
random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zsl_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
WINDOW_SIZE = 5
REQUESTS_PER_WINDOW = 50
FEATURE_COLS = ["mean_latency","max_latency","p95_latency","std_latency","error_rate","request_rate","http_500_count","cpu_percent","mem_usage_mb","mem_percent","io_read_bytes","io_write_bytes","net_rx_bytes","net_tx_bytes"]
CONTAINER_PROFILES = {
    "normal":       dict(cpu=(1.5,0.3),  mem=(118.0,3.0),  mem_pct=(11.8,0.3), io_r=(0,50),  io_w=(200,50),  net_rx=(5000,500), net_tx=(4000,400)),
    "error_spike":  dict(cpu=(2.5,0.5),  mem=(120.0,3.0),  mem_pct=(12.0,0.3), io_r=(0,50),  io_w=(300,60),  net_rx=(6000,600), net_tx=(5000,500)),
    "slow":         dict(cpu=(8.0,1.0),  mem=(115.0,3.0),  mem_pct=(11.5,0.3), io_r=(0,80),  io_w=(150,40),  net_rx=(4000,400), net_tx=(3000,300)),
    "memory_leak":  dict(cpu=(6.0,1.0),  mem=(210.0,10.0), mem_pct=(21.0,1.0), io_r=(0,200), io_w=(800,150), net_rx=(7000,750), net_tx=(6000,600)),
    "intermittent": dict(cpu=(3.0,2.0),  mem=(125.0,5.0),  mem_pct=(12.5,0.5), io_r=(0,200), io_w=(250,100), net_rx=(5500,800), net_tx=(4500,600)),
}
def gen_requests(fault_type, n=REQUESTS_PER_WINDOW):
    reqs = []
    for _ in range(n):
        if fault_type == "normal":
            lat = max(0.002, random.gauss(0.0075, 0.003)); st = 200
        elif fault_type == "error_spike":
            lat = max(0.002, random.gauss(0.0062, 0.003))
            st = 500 if random.random() < 0.30 else 200
        elif fault_type == "slow":
            lat = max(1.0, random.gauss(1.508, 0.015)); st = 200
        elif fault_type == "memory_leak":
            lat = max(0.03, random.gauss(0.12, 0.04))
            st = 503 if random.random() < 0.10 else 200
        elif fault_type == "intermittent":
            lat = max(0.002, random.gauss(0.40, 0.15)) if random.random() < 0.5 else max(0.002, random.gauss(0.05, 0.02))
            st = 500 if random.random() < 0.08 else 200
        reqs.append((lat, st))
    return reqs
def extract_api_features(reqs):
    lats = [r[0] for r in reqs]; n = len(lats)
    errors = [1 for r in reqs if r[1] != 200]
    sl = sorted(lats); mean_lat = sum(lats)/n; max_lat = max(lats)
    p95_lat = sl[max(0, int(math.ceil(0.95*n))-1)]
    std_lat = math.sqrt(sum((x-mean_lat)**2 for x in lats)/n)
    return [mean_lat, max_lat, p95_lat, std_lat, len(errors)/n, n/WINDOW_SIZE, len(errors)]
def extract_container_features(fault_type):
    p = CONTAINER_PROFILES[fault_type]
    return [max(0.0,random.gauss(*p["cpu"])),max(50.0,random.gauss(*p["mem"])),max(0.0,random.gauss(*p["mem_pct"])),max(0,int(random.gauss(*p["io_r"]))),max(0,int(random.gauss(*p["io_w"]))),max(0,int(random.gauss(*p["net_rx"]))),max(0,int(random.gauss(*p["net_tx"])))]
COUNTS = {"normal":60,"error_spike":40,"slow":40,"memory_leak":30,"intermittent":30}
if __name__ == "__main__":
    all_rows = []
    for ft, n in COUNTS.items():
        for _ in range(n):
            reqs = gen_requests(ft)
            row = extract_api_features(reqs) + extract_container_features(ft) + [ft]
            all_rows.append(row)
    with open(os.path.join(OUTPUT_DIR,"features.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_COLS + ["fault_type"])
        writer.writerows(all_rows)
    print(f"Generated {len(all_rows)} windows")
    for ft,n in COUNTS.items(): print(f"  {ft}: {n}")
"""
with open("zsl_module/generate_synthetic_log.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(gen)
print("✅ generate_synthetic_log.py written")

# ── 2. train.py — patch percentile 5 → 20 ───────────────────
txt = open("zsl_module/train.py", encoding="utf-8").read()
txt = txt.replace("np.percentile(sims, 5)", "np.percentile(sims, 20)")
with open("zsl_module/train.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(txt)
print("✅ train.py patched (percentile 20)")

# ── 3. zsl_server.py — fix +Inf query ────────────────────────
txt = open("zsl_module/zsl_server.py", encoding="utf-8").read()
txt = txt.replace("histogram_quantile(1.0,", "histogram_quantile(0.99,")
with open("zsl_module/zsl_server.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(txt)
print("✅ zsl_server.py patched (0.99 quantile)")

print()
print("All done. Now run:")
print("  docker compose down")
print("  docker rmi fault-tolerant-mlops-zsl --force")
print("  docker compose up -d --build")
