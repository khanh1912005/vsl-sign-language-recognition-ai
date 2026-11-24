"""
=============================================================
TRÍCH XUẤT ĐẶC TRƯNG HÌNH HỌC & ĐỘNG HỌC BẤT BIẾN
(INVARIANT SPATIAL-TEMPORAL FEATURE ENGINEERING)
=============================================================
Chuyển đổi chuỗi tọa độ thô MediaPipe (T, 258) thành biểu diễn bất biến
với vị trí đứng và khoảng cách camera, đồng thời làm nổi bật 4 thành tố
ngữ nghĩa của Ngôn ngữ ký hiệu (Handshape, Location, Orientation, Movement):

1. Upper-Body Pose chuẩn hóa theo tâm vai & độ rộng vai (25 khớp x 3 = 75)
2. Local Handshape chuẩn hóa theo cổ tay & kích thước bàn tay (2 tay x 20 x 3 = 120)
3. Global Hand-to-Body & Hand-to-Face (vị trí cổ tay + 5 đầu ngón so với
   tâm vai và mũi, chuẩn hóa theo độ rộng vai: 2 tay x 6 x 6 = 72)
4. Bimanual Interaction (tương quan 2 bàn tay: 6 điểm x 3 = 18)
5. Hand Presence Flags (cờ trạng thái xuất hiện tay trái/phải: 2)
6. Temporal Velocity (vận tốc chuyển động giữa các frame liên tiếp: 167)
-> Tổng cộng: 454 đặc trưng / frame.
=============================================================
"""

import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPPER_POSE_INDICES = list(range(25))  # 0..24 (bỏ qua chân/mắt cá 25..32 không xuất hiện trong khung hình)
FINGERTIP_AND_WRIST = [0, 4, 8, 12, 16, 20]  # Cổ tay + 5 đầu ngón tay


def extract_sequence_features(seq_raw):
    """
    Nhận vào chuỗi landmarks thô shape (T, 258) của 1 video,
    trả về chuỗi đặc trưng chuẩn hóa shape (T, 454).
    """
    T = seq_raw.shape[0]
    pose = seq_raw[:, :132].reshape(T, 33, 4)
    lh = seq_raw[:, 132:195].reshape(T, 21, 3)
    rh = seq_raw[:, 195:258].reshape(T, 21, 3)

    pose_xyz = pose[:, :, :3]  # (T, 33, 3)

    # Kiểm tra các frame có phát hiện pose, tay trái, tay phải
    pose_present = (np.abs(pose_xyz).sum(axis=(1, 2)) > 1e-5).astype(np.float32)
    lh_present = (np.abs(lh).sum(axis=(1, 2)) > 1e-5).astype(np.float32)
    rh_present = (np.abs(rh).sum(axis=(1, 2)) > 1e-5).astype(np.float32)

    # Nội suy tuyến tính cho các frame bị mất bàn tay tạm thời (nếu có >= 2 frame có tay)
    lh = _interpolate_missing_hand(lh, lh_present)
    rh = _interpolate_missing_hand(rh, rh_present)
    lh_valid = (np.abs(lh).sum(axis=(1, 2)) > 1e-5)
    rh_valid = (np.abs(rh).sum(axis=(1, 2)) > 1e-5)

    # 1. Tâm vai (Shoulder Center) & Độ rộng vai (Shoulder Scale)
    l_shoulder = pose_xyz[:, 11, :]  # (T, 3)
    r_shoulder = pose_xyz[:, 12, :]  # (T, 3)
    body_center = 0.5 * (l_shoulder + r_shoulder)  # (T, 3)
    nose = pose_xyz[:, 0, :]  # (T, 3)

    shoulder_dists = np.linalg.norm(l_shoulder[:, :2] - r_shoulder[:, :2], axis=-1)
    valid_dists = shoulder_dists[pose_present > 0.5]
    if len(valid_dists) > 0:
        body_scale = float(np.clip(np.median(valid_dists), 0.08, 0.8))
    else:
        body_scale = 0.25

    # Chuẩn hóa 25 điểm Upper-Body Pose theo tâm vai và body_scale
    upper_pose = pose_xyz[:, UPPER_POSE_INDICES, :]  # (T, 25, 3)
    norm_pose = (upper_pose - body_center[:, None, :]) / body_scale
    norm_pose[pose_present < 0.5] = 0.0
    norm_pose_flat = norm_pose.reshape(T, -1)  # (T, 75)

    # 2. Local Handshape (chuẩn hóa từng bàn tay theo cổ tay & kích thước lòng bàn tay)
    lh_shape = _compute_local_handshape(lh, lh_valid, body_scale)  # (T, 60)
    rh_shape = _compute_local_handshape(rh, rh_valid, body_scale)  # (T, 60)

    # 3. Global Hand Location (so với tâm vai và so với mũi)
    lh_key = lh[:, FINGERTIP_AND_WRIST, :]  # (T, 6, 3)
    rh_key = rh[:, FINGERTIP_AND_WRIST, :]  # (T, 6, 3)

    lh_rel_body = (lh_key - body_center[:, None, :]) / body_scale
    lh_rel_nose = (lh_key - nose[:, None, :]) / body_scale
    lh_rel_body[~lh_valid] = 0.0
    lh_rel_nose[~lh_valid] = 0.0

    rh_rel_body = (rh_key - body_center[:, None, :]) / body_scale
    rh_rel_nose = (rh_key - nose[:, None, :]) / body_scale
    rh_rel_body[~rh_valid] = 0.0
    rh_rel_nose[~rh_valid] = 0.0

    global_hand_loc = np.concatenate([
        lh_rel_body.reshape(T, -1),
        lh_rel_nose.reshape(T, -1),
        rh_rel_body.reshape(T, -1),
        rh_rel_nose.reshape(T, -1),
    ], axis=-1)  # (T, 72)

    # 4. Tương tác 2 bàn tay (Bimanual interaction)
    both_valid = lh_valid & rh_valid
    bimanual = (rh_key - lh_key) / body_scale  # (T, 6, 3)
    bimanual[~both_valid] = 0.0
    # Nếu thiếu bàn tay chi tiết, dùng khoảng cách 2 cổ tay từ Pose (khớp 15, 16)
    pose_wrists_diff = (pose_xyz[:, 16, :] - pose_xyz[:, 15, :]) / body_scale  # (T, 3)
    bimanual[:, 0, :] = np.where(both_valid[:, None], bimanual[:, 0, :], pose_wrists_diff)
    bimanual_flat = bimanual.reshape(T, -1)  # (T, 18)

    # 5. Cờ trạng thái bàn tay (2)
    flags = np.stack([lh_present, rh_present], axis=-1)  # (T, 2)

    # 6. Vận tốc chuyển động (Temporal Velocity) trên các đặc trưng vị trí & hình dạng
    kinematic_base = np.concatenate([
         global_hand_loc,   # 72
        bimanual_flat,     # 18
        lh_shape[:, :30],  # 30
        rh_shape[:, :30],  # 30
        norm_pose[:, [0, 11, 12, 13, 14, 15, 16], :].reshape(T, -1)  # 21 (mũi, vai, khuỷu, cổ tay)
    ], axis=-1)  # (T, 171)

    velocity = np.zeros_like(kinematic_base)
    velocity[1:] = kinematic_base[1:] - kinematic_base[:-1]

    features = np.concatenate([
        norm_pose_flat,    # 75
        lh_shape,          # 60
        rh_shape,          # 60
        global_hand_loc,   # 72
        bimanual_flat,     # 18
        flags,             # 2
        velocity,          # 171
    ], axis=-1).astype(np.float32)  # Tổng: 458 đặc trưng

    return np.clip(features, -10.0, 10.0)


def _interpolate_missing_hand(hand_seq, present_mask):
    """Nội suy tuyến tính cho các frame bị mất bàn tay ngắn hạn ở giữa video."""
    valid_idx = np.where(present_mask > 0.5)[0]
    if len(valid_idx) < 2 or len(valid_idx) == len(present_mask):
        return hand_seq

    out = hand_seq.copy()
    # Chỉ nội suy các khoảng trống ngắn (<= 5 frames) ở giữa chuỗi có tay
    for i in range(len(valid_idx) - 1):
        start_f = valid_idx[i]
        end_f = valid_idx[i + 1]
        gap = end_f - start_f
        if 1 < gap <= 6:
            for f in range(start_f + 1, end_f):
                alpha = (f - start_f) / float(gap)
                out[f] = (1.0 - alpha) * out[start_f] + alpha * out[end_f]
    return out


def _compute_local_handshape(hand_seq, valid_mask, fallback_scale):
    """Chuẩn hóa 20 khớp ngón tay theo cổ tay (khớp 0) và kích thước bàn tay."""
    T = hand_seq.shape[0]
    wrist = hand_seq[:, 0:1, :]  # (T, 1, 3)
    rel_joints = hand_seq[:, 1:, :] - wrist  # (T, 20, 3)

    # Kích thước bàn tay = khoảng cách từ cổ tay (0) đến khớp gốc ngón giữa (9)
    palm_sizes = np.linalg.norm(hand_seq[:, 9, :] - hand_seq[:, 0, :], axis=-1)
    valid_palms = palm_sizes[valid_mask]
    if len(valid_palms) > 0:
        hand_scale = float(np.clip(np.median(valid_palms), 0.03, 0.4))
    else:
        hand_scale = fallback_scale * 0.4

    norm_joints = rel_joints / hand_scale
    norm_joints[~valid_mask] = 0.0
    return norm_joints.reshape(T, -1)  # (T, 60)


def transform_dataset(X_raw):
    """Chuyển đổi toàn bộ tập dữ liệu (N, T, 258) -> (N, T, 458)."""
    return np.stack([extract_sequence_features(X_raw[i]) for i in range(len(X_raw))], axis=0)


if __name__ == "__main__":
    import os, time
    PROCESSED_DIR = rf"{BASE_DIR}\processed_data"
    X_raw = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    t0 = time.time()
    X_feat = transform_dataset(X_raw)
    dt = time.time() - t0
    print(f"Transformed {X_raw.shape} -> {X_feat.shape} in {dt:.2f}s ({dt/len(X_raw)*1000:.2f} ms/video)")
    print(f"Min: {X_feat.min():.3f}, Max: {X_feat.max():.3f}, Mean: {X_feat.mean():.4f}, Std: {X_feat.std():.4f}")
    np.save(os.path.join(PROCESSED_DIR, "X_features.npy"), X_feat)
    print("Saved X_features.npy successfully!")
