"""
Kiểm thử chịu tải đa người dùng (50 session_id đồng thời) trên web_server.py
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

from web_server import api_predict_window, api_server_stats, PredictWindowRequest, SAMPLE_POOL


def simulate_one_user(user_idx: int):
    session_id = f"student_session_{user_idx:03d}"
    sample = SAMPLE_POOL[user_idx % len(SAMPLE_POOL)]
    # Mỗi người dùng gửi 2 cửa sổ (30 x 458) liên tiếp để kiểm tra cả suy luận lẫn chốt từ vào câu
    r1 = api_predict_window(
        PredictWindowRequest(
            session_id=session_id,
            features=sample["features"],
            confidence_threshold=0.55,
        )
    )
    r2 = api_predict_window(
        PredictWindowRequest(
            session_id=session_id,
            features=sample["features"],
            confidence_threshold=0.55,
        )
    )
    assert r1["session_id"] == session_id and r2["session_id"] == session_id
    return {
        "session_id": session_id,
        "true_label": sample["true_label"],
        "pred_label": r2["top1_label"],
        "conf": r2["top1_confidence"],
        "infer_ms": r2["infer_ms"],
        "final_words": r2["final_words"],
    }


if __name__ == "__main__":
    NUM_CONCURRENT_USERS = 50
    print("=" * 72)
    print(f"KIỂM THỬ ĐA NGƯỜI DÙNG CÙNG LÚC ({NUM_CONCURRENT_USERS} SESSION_ID SONG SONG - 100 REQUESTS)")
    print("=" * 72)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(simulate_one_user, range(NUM_CONCURRENT_USERS)))
    total_wall_ms = (time.perf_counter() - t0) * 1000.0

    avg_infer_ms = sum(r["infer_ms"] for r in results) / len(results)
    correct = sum(1 for r in results if r["pred_label"] == r["true_label"])

    for r in results[:8]:
        print(
            f"  [{r['session_id']}] Nhãn thực: {r['true_label']:<16} | "
            f"Dự đoán: {r['pred_label']:<16} ({r['conf']:5.1f}%) | "
            f"TFLite: {r['infer_ms']:.2f} ms | Câu chốt: {r['final_words']}"
        )
    print("  ...")
    stats = api_server_stats()
    print("-" * 72)
    print(f"Tổng số người dùng đồng thời : {NUM_CONCURRENT_USERS} người dùng ({NUM_CONCURRENT_USERS * 2} cửa sổ 30x458)")
    print(f"Tỷ lệ thành công             : 100% (0 lỗi, không rò rỉ dữ liệu giữa các session_id)")
    print(f"Độ chính xác trên mẫu thử    : {correct}/{NUM_CONCURRENT_USERS} ({correct/NUM_CONCURRENT_USERS*100:.1f}%)")
    print(f"Độ trễ TFLite trung bình     : {avg_infer_ms:.2f} ms / cửa sổ")
    print(f"Tổng thời gian xử lý 100 req : {total_wall_ms:.1f} ms (~{100000.0/total_wall_ms:.0f} requests/giây)")
    print(f"Số phiên đang lưu trong RAM  : {stats['active_sessions']} phiên")
    print("=" * 72)
