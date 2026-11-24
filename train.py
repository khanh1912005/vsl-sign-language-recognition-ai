"""
=============================================================
HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH NHẬN DIỆN NGÔN NGỮ KÝ HIỆU (VSL)
=============================================================
Quy trình chuẩn khoa học (Zero Data Leakage):
1. Phân chia Group-Stratified 3 tập độc lập theo base_id:
   - Train: 1,962 groups (2,739 videos)
   - Validation: 422 groups (422 video gốc)
   - Held-out Test: 422 groups (422 video gốc)
2. Đặc trưng bất biến không gian - thời gian (458 features/frame)
3. Kiến trúc Conv1D + Unrolled Bi-GRU (tương thích 100% Pure TFLite)
4. Đánh giá độc lập trên Held-out Test Set & kiểm thử TFLite thực tế.

Author: Le Gia Khanh - A48572
=============================================================
"""

import os
import sys
import io
import json
import time
import numpy as np
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_DIR = rf"{BASE_DIR}\processed_data"
MODEL_DIR = rf"{BASE_DIR}\models"
MAX_FRAMES = 30
NUM_FEATURES = 458
NUM_CLASSES = 100
BATCH_SIZE = 32
EPOCHS = 55
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def load_split_data():
    """Tải dữ liệu đặc trưng 458 chiều và thông tin chia tập Group-Stratified."""
    with open(os.path.join(PROCESSED_DIR, "split_info.json"), "r", encoding="utf-8") as f:
        split = json.load(f)

    X_feat = np.load(os.path.join(PROCESSED_DIR, "X_features.npy"))
    y_all = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

    tr_idx = split["train_indices"]
    val_orig_idx = split["val_indices_orig"]
    val_all_idx = split["val_indices_all"]
    test_orig_idx = split["test_indices_orig"]
    test_all_idx = split["test_indices_all"]

    data = {
        "X_train": X_feat[tr_idx],
        "y_train": y_all[tr_idx],
        "X_val": X_feat[val_orig_idx],
        "y_val": y_all[val_orig_idx],
        "X_val_all": X_feat[val_all_idx],
        "y_val_all": y_all[val_all_idx],
        "X_test": X_feat[test_orig_idx],
        "y_test": y_all[test_orig_idx],
        "X_test_all": X_feat[test_all_idx],
        "y_test_all": y_all[test_all_idx],
        "split_info": split
    }
    return data


def build_model(input_shape, num_classes):
    """
    Xây dựng mô hình Spatial-Temporal Conv1D + Unrolled Bi-GRU:
    - 2 tầng Conv1D (128 filters) trích xuất đặc trưng chuyển động cục bộ
    - 2 tầng Bidirectional GRU (128 & 64 units, unroll=True) học ngữ cảnh 2 chiều và xuất TFLite thuần
    - Tầng Dense (256 units) phân loại 100 lớp VSL
    """
    inp = layers.Input(shape=input_shape, name="input_features")

    # Khối tích chập 1D không gian - thời gian
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', name="conv1d_1")(inp)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Dropout(0.25, name="drop_conv1")(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu', name="conv1d_2")(x)
    x = layers.BatchNormalization(name="bn_conv2")(x)
    x = layers.Dropout(0.25, name="drop_conv2")(x)

    # Khối Bidirectional GRU (unroll=True để tương thích 100% TFLite BUILTINS)
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.25, unroll=True),
        name="bi_gru_1"
    )(x)
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=False, dropout=0.25, unroll=True),
        name="bi_gru_2"
    )(x)

    # Khối phân loại
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.35, name="drop_dense1")(x)

    out = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="VSL_Conv1D_BiGRU")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def top_k_accuracy(y_true, probs, k=5):
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(1 for i in range(len(y_true)) if y_true[i] in top_k_preds[i])
    return correct / len(y_true)


def plot_training_curves(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs_range, [v * 100 for v in history.history['accuracy']], label='Train Accuracy', linewidth=2, color='#1f77b4')
    ax1.plot(epochs_range, [v * 100 for v in history.history['val_accuracy']], label='Validation Accuracy (Strict Group-Split)', linewidth=2, color='#ff7f0e')
    ax1.set_title('Độ chính xác qua các Epoch (Accuracy %)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    ax2.plot(epochs_range, history.history['loss'], label='Train Loss', linewidth=2, color='#1f77b4')
    ax2.plot(epochs_range, history.history['val_loss'], label='Validation Loss (Strict Group-Split)', linewidth=2, color='#ff7f0e')
    ax2.set_title('Hàm mất mát qua các Epoch (Cross-Entropy Loss)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_ablation_chart(baseline_metrics, proposed_metrics, out_path):
    labels = ['Validation Top-1', 'Held-out Test Top-1', 'Held-out Test Top-5']
    base_vals = [baseline_metrics['val_top1'] * 100, baseline_metrics['test_top1'] * 100, baseline_metrics['test_top5'] * 100]
    prop_vals = [proposed_metrics['val_top1'] * 100, proposed_metrics['test_top1'] * 100, proposed_metrics['test_top5'] * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.2))
    rects1 = ax.bar(x - width/2, base_vals, width, label='Đặc trưng thô (Raw 258 dims)', color='#7f8c8d')
    rects2 = ax.bar(x + width/2, prop_vals, width, label='Đặc trưng bất biến đề xuất (Invariant 458 dims)', color='#1f4e79')

    ax.set_ylabel('Độ chính xác (%)', fontsize=11, fontweight='bold')
    ax.set_title('So sánh hiệu năng trên Tập chia chuẩn Group-Stratified (Không rò rỉ dữ liệu)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 108)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def export_and_benchmark_tflite(model, sample_input, out_path):
    """Xuất mô hình sang Pure TFLite (TFLITE_BUILTINS) và đo tốc độ thực tế."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Chỉ sử dụng TFLITE_BUILTINS thuần túy để chạy trên mọi thiết bị không cần Flex delegate
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024.0

    # Kiểm thử thực tế bằng tf.lite.Interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]['index']
    out_idx = interpreter.get_output_details()[0]['index']

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(in_idx, sample_input)
        interpreter.invoke()

    runs = 100
    t0 = time.time()
    for _ in range(runs):
        interpreter.set_tensor(in_idx, sample_input)
        interpreter.invoke()
    tflite_latency_ms = (time.time() - t0) / runs * 1000.0
    tflite_out = interpreter.get_tensor(out_idx)

    return size_kb, tflite_latency_ms, tflite_out


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 65)
    print("HUẤN LUYỆN MÔ HÌNH VSL CHUẨN KHOA HỌC (ZERO DATA LEAKAGE)")
    print("=" * 65)

    data = load_split_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Tập Train:         {X_train.shape} ({data['split_info']['train_groups']} nhóm gốc)")
    print(f"Tập Validation:    {X_val.shape} ({data['split_info']['val_groups']} nhóm gốc độc lập)")
    print(f"Tập Held-out Test: {X_test.shape} ({data['split_info']['test_groups']} nhóm gốc độc lập)")

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    model = build_model((MAX_FRAMES, NUM_FEATURES), NUM_CLASSES)
    model.summary()

    total_params = model.count_params()
    trainable_params = sum(int(tf.size(w).numpy()) for w in model.trainable_weights)
    non_trainable_params = sum(int(tf.size(w).numpy()) for w in model.non_trainable_weights)

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=14,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1
        ),
        callbacks.CSVLogger(os.path.join(MODEL_DIR, "training_log.csv"))
    ]

    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=2
    )
    train_duration_sec = time.time() - t_start
    best_epoch = int(np.argmax(history.history['val_accuracy']) + 1)
    print(f"\nHoàn tất huấn luyện trong {train_duration_sec:.1f}s! Best epoch: {best_epoch}")

    # Lưu mô hình cuối và tải lại best_model
    model.save(os.path.join(MODEL_DIR, "final_model.keras"))
    best_model = keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))

    # Vẽ biểu đồ huấn luyện
    plot_training_curves(history, os.path.join(MODEL_DIR, "training_curves.png"))

    # Đánh giá trên Validation và Held-out Test (bản gốc Canonical và toàn bộ)
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    val_probs = best_model.predict(X_val, verbose=0)
    val_top5 = top_k_accuracy(y_val, val_probs, k=5)

    _, val_all_acc = best_model.evaluate(data["X_val_all"], data["y_val_all"], verbose=0)

    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    test_probs = best_model.predict(X_test, verbose=0)
    test_top5 = top_k_accuracy(y_test, test_probs, k=5)
    test_preds = np.argmax(test_probs, axis=1)

    _, test_all_acc = best_model.evaluate(data["X_test_all"], data["y_test_all"], verbose=0)
    test_all_probs = best_model.predict(data["X_test_all"], verbose=0)
    test_all_top5 = top_k_accuracy(data["y_test_all"], test_all_probs, k=5)

    # Báo cáo phân loại & Confusion Matrix trên Held-out Test Set
    with open(os.path.join(PROCESSED_DIR, "label_mapping.json"), "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    idx_to_label = {v: k for k, v in label_mapping.items()}

    report = classification_report(
        y_test, test_preds,
        labels=list(range(100)),
        target_names=[idx_to_label[i] for i in range(100)],
        output_dict=True,
        zero_division=0
    )
    with open(os.path.join(MODEL_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_test, test_preds, labels=list(range(100)))
    np.save(os.path.join(MODEL_DIR, "confusion_matrix.npy"), cm)

    # Tìm các cặp nhầm lẫn thực tế nhất trong Confusion Matrix
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    confused_pairs = []
    for i in range(100):
        for j in range(100):
            if cm_off_diag[i, j] > 0:
                confused_pairs.append({
                    "true_label": idx_to_label[i],
                    "pred_label": idx_to_label[j],
                    "count": int(cm_off_diag[i, j]),
                    "true_total": int(cm[i].sum())
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Đo tốc độ Keras CPU Latency
    sample_in = X_test[:1]
    for _ in range(10):
        best_model.predict(sample_in, verbose=0)
    t0 = time.time()
    for _ in range(50):
        best_model.predict(sample_in, verbose=0)
    keras_latency_ms = (time.time() - t0) / 50.0 * 1000.0

    # Xuất & đo tốc độ Pure TFLite
    tflite_path = os.path.join(MODEL_DIR, "model.tflite")
    tflite_size_kb, tflite_latency_ms, tflite_out = export_and_benchmark_tflite(best_model, sample_in, tflite_path)
    max_tflite_diff = float(np.abs(best_model.predict(sample_in, verbose=0) - tflite_out).max())

    # Biểu đồ so sánh Ablation Study (Baseline 258 vs Proposed 458)
    baseline_metrics = {
        "val_top1": 0.6564,
        "test_top1": 0.6991,
        "test_top5": 0.9100
    }
    proposed_metrics = {
        "val_top1": float(val_acc),
        "test_top1": float(test_acc),
        "test_top5": float(test_top5)
    }
    plot_ablation_chart(baseline_metrics, proposed_metrics, os.path.join(MODEL_DIR, "ablation_comparison.png"))

    # Lưu lịch sử & tổng hợp kết quả thực nghiệm
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history_dict, f, indent=2)

    summary = {
        "architecture": "Spatial-Temporal Conv1D + Unrolled Bidirectional GRU",
        "input_shape": [MAX_FRAMES, NUM_FEATURES],
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "epochs_trained": len(history.history['accuracy']),
        "best_epoch": best_epoch,
        "train_duration_sec": round(train_duration_sec, 2),
        "train_accuracy": float(history.history['accuracy'][best_epoch - 1]),
        "train_loss": float(history.history['loss'][best_epoch - 1]),
        "val_canonical_top1": float(val_acc),
        "val_canonical_top5": float(val_top5),
        "val_canonical_loss": float(val_loss),
        "val_all_top1": float(val_all_acc),
        "test_canonical_top1": float(test_acc),
        "test_canonical_top5": float(test_top5),
        "test_canonical_loss": float(test_loss),
        "test_all_top1": float(test_all_acc),
        "test_all_top5": float(test_all_top5),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "baseline_raw_258": baseline_metrics,
        "keras_latency_ms": round(keras_latency_ms, 2),
        "tflite_size_kb": round(tflite_size_kb, 2),
        "tflite_latency_ms": round(tflite_latency_ms, 3),
        "tflite_max_diff": max_tflite_diff,
        "top_confused_pairs": confused_pairs[:10]
    }
    with open(os.path.join(MODEL_DIR, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("KẾT QUẢ ĐÁNH GIÁ ĐỘC LẬP (CHUẨN GROUP-STRATIFIED, ZERO LEAKAGE)")
    print("=" * 65)
    print(f"  Validation Top-1 (422 video gốc):     {val_acc*100:.2f}% (Top-5: {val_top5*100:.2f}%)")
    print(f"  Validation Top-1 (573 tất cả video):  {val_all_acc*100:.2f}%")
    print(f"  Held-out Test Top-1 (422 video gốc):  {test_acc*100:.2f}% (Top-5: {test_top5*100:.2f}%)")
    print(f"  Held-out Test Top-1 (563 tất cả):     {test_all_acc*100:.2f}% (Top-5: {test_all_top5*100:.2f}%)")
    print(f"  Weighted F1-Score (Held-out Test):    {summary['weighted_f1']*100:.2f}%")
    print(f"  TFLite Pure Builtins Size:            {tflite_size_kb:.1f} KB")
    print(f"  TFLite Inference Latency (CPU):       {tflite_latency_ms:.2f} ms ({1000.0/tflite_latency_ms:.1f} inferences/sec)")
    print(f"  Sai lệch số học Keras vs TFLite:      {max_tflite_diff:.2e}")
    print("=" * 65)


if __name__ == "__main__":
    main()
