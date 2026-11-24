# -*- coding: utf-8 -*-
"""
ĐO HIỆU NĂNG MÁY CHỦ WEB ĐA NGƯỜI DÙNG QUA HTTP THẬT (phục vụ báo cáo)
- Gửi đồng thời N người dùng (mỗi người 1 session_id, 2 cửa sổ 30x458) tới /api/predict_window
- Đo độ trễ đầu-cuối (p50/p95), thông lượng (req/s), độ trễ TFLite phía server, tỷ lệ lỗi.
- Xuất models/web_benchmark.json và models/web_benchmark.png
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://127.0.0.1:8000"
OUT_DIR = rf"{BASE_DIR}\models"


def http_json(url, payload=None, timeout=60):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


pool = http_json(f"{BASE}/api/sample_pool?count=24")["samples"]
print(f"Đã tải {len(pool)} mẫu 30x458")

# Warm-up
for s in pool[:5]:
    http_json(f"{BASE}/api/predict_window", {"session_id": "warmup", "features": s["features"], "is_isolated_video": True})


def one_user(uid, level):
    sid = f"bench_{level}_{uid}"
    results = []
    for k in range(2):
        s = pool[(uid + k) % len(pool)]
        t0 = time.perf_counter()
        try:
            res = http_json(f"{BASE}/api/predict_window",
                            {"session_id": sid, "features": s["features"],
                             "confidence_threshold": 0.55, "is_isolated_video": True})
            dt = (time.perf_counter() - t0) * 1000
            results.append({"ok": True, "e2e_ms": dt, "infer_ms": res["infer_ms"],
                            "correct": res["top1_label"] == s["true_label"]})
        except Exception as e:
            results.append({"ok": False, "e2e_ms": (time.perf_counter() - t0) * 1000, "err": str(e)})
    return results


levels = [1, 10, 25, 50, 100]
summary = []
for n in levels:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        all_res = [r for lst in ex.map(lambda u: one_user(u, n), range(n)) for r in lst]
    wall = time.perf_counter() - t0
    ok = [r for r in all_res if r["ok"]]
    e2e = np.array([r["e2e_ms"] for r in ok])
    inf = np.array([r["infer_ms"] for r in ok])
    row = {
        "users": n,
        "requests": len(all_res),
        "errors": len(all_res) - len(ok),
        "success_rate": round(100.0 * len(ok) / len(all_res), 2),
        "throughput_rps": round(len(ok) / wall, 1),
        "wall_ms": round(wall * 1000, 1),
        "e2e_p50_ms": round(float(np.percentile(e2e, 50)), 1),
        "e2e_p95_ms": round(float(np.percentile(e2e, 95)), 1),
        "tflite_avg_ms": round(float(inf.mean()), 2),
        "accuracy": round(100.0 * sum(r["correct"] for r in ok) / len(ok), 2),
    }
    summary.append(row)
    print(row)

with open(os.path.join(OUT_DIR, "web_benchmark.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Biểu đồ
users = [r["users"] for r in summary]
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar([str(u) for u in users], [r["throughput_rps"] for r in summary], color="#1f4e79", label="Thông lượng (requests/giây)")
ax1.set_xlabel("Số người dùng gửi đồng thời", fontsize=11, fontweight="bold")
ax1.set_ylabel("Thông lượng (requests/giây)", fontsize=11, fontweight="bold", color="#1f4e79")
for i, r in enumerate(summary):
    ax1.annotate(f'{r["throughput_rps"]}', (i, r["throughput_rps"]), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2 = ax1.twinx()
ax2.plot([str(u) for u in users], [r["e2e_p95_ms"] for r in summary], "o-", color="#e67e22", linewidth=2.2, label="Độ trễ đầu-cuối p95 (ms)")
ax2.plot([str(u) for u in users], [r["tflite_avg_ms"] for r in summary], "s--", color="#27ae60", linewidth=2, label="Độ trễ TFLite trung bình (ms)")
ax2.set_ylabel("Độ trễ (ms)", fontsize=11, fontweight="bold", color="#e67e22")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
plt.title("Hiệu năng máy chủ Web đa người dùng (Singleton TFLite + session_id)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "web_benchmark.png"), dpi=200)
print("Đã lưu models/web_benchmark.json và models/web_benchmark.png")
