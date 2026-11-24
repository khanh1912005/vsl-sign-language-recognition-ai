"""
=============================================================
CHIA TẬP DỮ LIỆU CHUẨN KHOA HỌC (GROUP-STRATIFIED 3-WAY SPLIT)
=============================================================
Đảm bảo:
1. Không rò rỉ dữ liệu (Zero Data Leakage): Tất cả các biến thể (_1, _2, _3)
   của cùng một video gốc (base_id) bắt buộc nằm trong cùng một tập.
2. Phân tầng đầy đủ (100% Class Coverage): Cả 100 lớp đều có mặt trong
   Train, Validation và Held-out Test.
3. Tập Test độc lập (Held-out Test): Chỉ dùng đúng 1 lần để đánh giá cuối cùng.
=============================================================
"""

import os
import sys
import io
import json
import numpy as np
from collections import defaultdict
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROCESSED_DIR = rf"{BASE_DIR}\processed_data"
RANDOM_SEED = 42


def create_group_stratified_split():
    with open(os.path.join(PROCESSED_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    X_all = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_all = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

    assert len(metadata) == len(X_all) == len(y_all), "Kích thước metadata và X_train không khớp!"

    # Nhóm các index theo (label_idx, base_id)
    class_to_groups = defaultdict(lambda: defaultdict(list))
    for idx, item in enumerate(metadata):
        c = item["label_idx"]
        bid = item["base_id"]
        class_to_groups[c][bid].append(idx)

    rng = np.random.default_rng(RANDOM_SEED)

    train_indices = []
    val_indices_all = []
    val_indices_orig = []
    test_indices_all = []
    test_indices_orig = []

    train_groups = set()
    val_groups = set()
    test_groups = set()

    for c in range(100):
        groups_dict = class_to_groups[c]
        base_ids = sorted(groups_dict.keys())
        rng.shuffle(base_ids)

        n_groups = len(base_ids)
        # Tối thiểu 1 group cho test, 1 group cho val, phần còn lại (>=3) cho train
        n_test = max(1, int(round(0.15 * n_groups)))
        n_val = max(1, int(round(0.15 * n_groups)))
        if n_groups - n_test - n_val < 3:
            n_test = 1
            n_val = 1

        test_bids = base_ids[:n_test]
        val_bids = base_ids[n_test:n_test + n_val]
        train_bids = base_ids[n_test + n_val:]

        for bid in train_bids:
            train_groups.add(bid)
            train_indices.extend(groups_dict[bid])

        for bid in val_bids:
            val_groups.add(bid)
            for idx in groups_dict[bid]:
                val_indices_all.append(idx)
                if not metadata[idx]["is_augmented"]:
                    val_indices_orig.append(idx)

        for bid in test_bids:
            test_groups.add(bid)
            for idx in groups_dict[bid]:
                test_indices_all.append(idx)
                if not metadata[idx]["is_augmented"]:
                    test_indices_orig.append(idx)

    # Kiểm tra chặt chẽ không có bất kỳ giao thoa nào
    assert len(train_groups & val_groups) == 0, "Rò rỉ giữa Train và Val!"
    assert len(train_groups & test_groups) == 0, "Rò rỉ giữa Train và Test!"
    assert len(val_groups & test_groups) == 0, "Rò rỉ giữa Val và Test!"

    split_info = {
        "seed": RANDOM_SEED,
        "total_videos": len(metadata),
        "total_unique_groups": len(train_groups) + len(val_groups) + len(test_groups),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_samples": len(train_indices),
        "val_samples_all": len(val_indices_all),
        "val_samples_orig": len(val_indices_orig),
        "test_samples_all": len(test_indices_all),
        "test_samples_orig": len(test_indices_orig),
        "train_indices": train_indices,
        "val_indices_all": val_indices_all,
        "val_indices_orig": val_indices_orig,
        "test_indices_all": test_indices_all,
        "test_indices_orig": test_indices_orig,
    }

    with open(os.path.join(PROCESSED_DIR, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print("=" * 60)
    print("THỐNG KÊ PHÂN CHIA TẬP DỮ LIỆU (GROUP-STRATIFIED SPLIT)")
    print("=" * 60)
    print(f"Tổng số video:              {split_info['total_videos']}")
    print(f"Tổng số video gốc (groups): {split_info['total_unique_groups']}")
    print(f"Kiểm tra giao thoa (leak):  0 (Hoàn toàn độc lập)")
    print("-" * 60)
    print(f"1. Tập Train:     {split_info['train_groups']:4d} groups | {split_info['train_samples']:4d} videos (gồm cả bản tăng cường)")
    print(f"2. Tập Val:       {split_info['val_groups']:4d} groups | {split_info['val_samples_orig']:4d} video gốc ({split_info['val_samples_all']} nếu tính cả biến thể)")
    print(f"3. Tập Test:      {split_info['test_groups']:4d} groups | {split_info['test_samples_orig']:4d} video gốc ({split_info['test_samples_all']} nếu tính cả biến thể)")
    print("-" * 60)
    print(f"Số lớp trong Train: {len(np.unique(y_all[train_indices]))}/100")
    print(f"Số lớp trong Val:   {len(np.unique(y_all[val_indices_orig]))}/100")
    print(f"Số lớp trong Test:  {len(np.unique(y_all[test_indices_orig]))}/100")
    print("=" * 60)


if __name__ == "__main__":
    create_group_stratified_split()
