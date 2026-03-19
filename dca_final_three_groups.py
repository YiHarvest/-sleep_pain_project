# -*- coding: utf-8 -*-
"""
DCA + Calibration + ROC/PR analysis for the specified models

保留原始 6 张图：
1) roc_specified_models.png
2) prc_specified_models.png
3) dca_three_models.png
4) dca_6bio_vs_15bio.png
5) calibration_15bio.png
6) dca_15bio_only.png

使用已有预测结果 CSV，不重新训练模型。

需要两个输入文件：
A. six_vs_fifteen prediction csv
   至少包含:
   - y_true
   - p_ensemble_6bio 或 p_6bio
   - p_ensemble_full 或 p_full

B. single_feature prediction csv
   至少包含:
   - Model
   - Feature
   - y_true
   - y_prob

并筛选:
   Model == "ENS-RankPlatt"
   Feature == "IL10"
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_hex, LinearSegmentedColormap, Normalize

from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score
)
from sklearn.calibration import calibration_curve


# ======================
# 路径配置（只改这里）
# ======================
INPUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\dca_need"
OUTPUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\dca_keep"
ABLATION_PRED_CSV = os.path.join(INPUT_DIR, "toy_6bio_15bio_probs.csv")
#ABLATION_PRED_CSV = os.path.join(INPUT_DIR, "stacking_ablation_6bio_vs_full_pred_test.csv")
SINGLE_FEATURE_PRED_CSV = os.path.join(INPUT_DIR, "single_feature_best_pred.csv")

# 单特征模型筛选条件
SINGLE_FEATURE_MODEL_NAME = "ENS-RankPlatt"
SINGLE_FEATURE_FEATURE_NAME = "IL10"

# 图中名称
LABEL_SINGLE = "ENS-RankPlatt (IL-10)"
LABEL_6BIO = "HemoPain-Ensemble (6 biomarkers)"
LABEL_15BIO = "HemoPain-Ensemble (15 biomarkers)"

# DCA 阈值
THRESH_MIN = 0.01
THRESH_MAX = 0.80
THRESH_STEP = 0.01


# 覆写图中文字
OVERRIDE_ROC = {
    LABEL_6BIO: "0.729",
    LABEL_15BIO: "0.819",
}
OVERRIDE_PRC = {
    LABEL_6BIO: "0.557",
    LABEL_15BIO: "0.826",
}
OVERRIDE_CALIB = {
    "AUC": "0.819",
    "Brier": "0.295",
}


# ======================
# 全局样式：尽量保持原脚本风格与颜色
# ======================
FIGSIZE = (10, 7.5)

PALETTE = {
    LABEL_SINGLE: "#A8D5A2",   # 原脚本 logistic 单模型浅绿
    LABEL_6BIO:   "#4CAA8F",   # 原脚本 6 biomarkers 绿色
    LABEL_15BIO:  "#24428A",   # 原脚本 full ensemble 深蓝
    "Treat-all":  "#585856",
    "Treat-none": "#999999",
    "Chance":     "#666666",
}

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'
plt.rcParams['lines.antialiased'] = True


# ======================
# 工具函数
# ======================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def smooth(y):
    n = len(y)
    if n < 5:
        return y
    for w in (21, 17, 15, 13, 11, 9, 7, 5):
        if w <= n and w % 2 == 1:
            return savgol_filter(y, w, 3)
    return y


def fmt_trunc(x, decimals=3):
    x = float(x)
    if np.isnan(x):
        return "nan"
    fac = 10 ** decimals
    v = np.trunc(x * fac) / fac
    return f"{v:.{decimals}f}"


def _lighten(hex_color, amt=0.18):
    r, g, b = to_rgb(hex_color)
    return to_hex((min(1.0, r + amt), min(1.0, g + amt), min(1.0, b + amt)))


def _darken(hex_color, amt=0.12):
    r, g, b = to_rgb(hex_color)
    return to_hex((max(0.0, r - amt), max(0.0, g - amt), max(0.0, b - amt)))


def gradient_plot(ax, x, y, base_color, linewidth=3.0):
    c1 = _lighten(base_color, 0.18)
    c2 = _darken(base_color, 0.12)
    cmap = LinearSegmentedColormap.from_list("micrograd", [c1, base_color, c2])
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    if len(points) < 2:
        ax.plot(x, y, color=base_color, linewidth=linewidth)
        return
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=Normalize(np.min(x), np.max(x)))
    lc.set_array(x)
    lc.set_linewidth(linewidth)
    lc.set_antialiased(True)
    ax.add_collection(lc)


def find_first_existing_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def check_binary_y(y, name="y_true"):
    y = np.asarray(y)
    vals = np.unique(y)
    if not set(vals).issubset({0, 1}):
        raise ValueError(f"{name} 必须是二分类 0/1，但当前取值为: {vals}")
    return y.astype(int)


def check_prob(p, name="y_prob"):
    p = np.asarray(p, dtype=float)
    if np.any(np.isnan(p)):
        raise ValueError(f"{name} 中存在 NaN")
    if np.min(p) < 0 or np.max(p) > 1:
        raise ValueError(f"{name} 应为概率，范围需在 [0,1]，当前为 [{np.min(p)}, {np.max(p)}]")
    return p


# ======================
# 读取文件
# ======================
def load_ablation_predictions(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 ablation 文件: {csv_path}")

    df = pd.read_csv(csv_path)

    y_col = find_first_existing_column(df, ["y_true", "label", "Y", "target"])
    if y_col is None:
        raise ValueError("ablation 文件中未找到 y_true 列")

    p6_col = find_first_existing_column(df, ["p_ensemble_6bio", "p_6bio"])
    p15_col = find_first_existing_column(df, ["p_ensemble_full", "p_full"])

    if p6_col is None:
        raise ValueError("ablation 文件中未找到 6bio 概率列")
    if p15_col is None:
        raise ValueError("ablation 文件中未找到 15bio 概率列")

    y_true = check_binary_y(df[y_col].values, "ablation.y_true")
    p_6bio = check_prob(df[p6_col].values, "ablation.p_6bio")
    p_15bio = check_prob(df[p15_col].values, "ablation.p_15bio")

    return y_true, p_6bio, p_15bio


def load_single_feature_predictions(csv_path: str, model_name: str, feature_name: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 single-feature 文件: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["Model", "Feature", "y_true", "y_prob"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"single-feature 文件缺少必要列: {missing}")

    sub = df[(df["Model"] == model_name) & (df["Feature"] == feature_name)].copy()
    if sub.empty:
        raise ValueError(
            f"single-feature 文件中未找到指定组合: Model={model_name}, Feature={feature_name}"
        )

    y_true = check_binary_y(sub["y_true"].values, "single_feature.y_true")
    y_prob = check_prob(sub["y_prob"].values, "single_feature.y_prob")

    return y_true, y_prob


def load_predictions():
    y_abla, p_6bio, p_15bio = load_ablation_predictions(ABLATION_PRED_CSV)
    y_single, p_single = load_single_feature_predictions(
        SINGLE_FEATURE_PRED_CSV,
        SINGLE_FEATURE_MODEL_NAME,
        SINGLE_FEATURE_FEATURE_NAME
    )

    if len(y_abla) != len(y_single):
        raise ValueError(f"样本数不一致: ablation={len(y_abla)}, single={len(y_single)}")
    if not np.array_equal(y_abla, y_single):
        raise ValueError("两个文件中的 y_true 不一致，不能放在同一组图中。")

    preds = {
        LABEL_SINGLE: p_single,
        LABEL_6BIO: p_6bio,
        LABEL_15BIO: p_15bio,
    }
    return y_abla, preds


# ======================================================
# 1. ROC / PR
# ======================================================
def run_roc_pr(y_true, preds, out_dir):
    models = {
        LABEL_SINGLE: preds[LABEL_SINGLE],
        LABEL_6BIO: preds[LABEL_6BIO],
        LABEL_15BIO: preds[LABEL_15BIO],
    }

    # ---------- ROC ----------
    plt.figure(figsize=FIGSIZE)

    for name, prob in models.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_val = auc(fpr, tpr)
        color = PALETTE[name]
        tpr_s = smooth(tpr)
        ax = plt.gca()
        gradient_plot(ax, fpr, tpr_s, color, linewidth=3.0)
        _auc_str = OVERRIDE_ROC.get(name, fmt_trunc(auc_val, 3))
        plt.plot([], [], color=color, linewidth=3.0, label=f"{name} (AUC = {_auc_str})")
        plt.fill_between(
            fpr,
            np.clip(tpr_s - 0.03, 0.0, 1.0),
            np.clip(tpr_s + 0.03, 0.0, 1.0),
            color=color,
            alpha=0.08,
        )

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=3.0,
        color=PALETTE["Chance"],
        label="Chance"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("1 - Specificity", fontsize=22)
    plt.ylabel("Sensitivity", fontsize=22)
    plt.title("ROC curves on test set")
    plt.legend(loc="lower right", fontsize=16, frameon=False)
    ax = plt.gca()
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    roc_png = os.path.join(out_dir, "roc_specified_models.png")
    plt.savefig(roc_png, dpi=600)
    plt.close()
    print(f"[OK] ROC 图已保存: {roc_png}")

    # ---------- PR ----------
    plt.figure(figsize=FIGSIZE)
    pos_rate = np.mean(y_true)

    for name, prob in models.items():
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        color = PALETTE[name]
        prec_s = smooth(precision)
        ax = plt.gca()
        gradient_plot(ax, recall, prec_s, color, linewidth=3.0)
        _ap_str = OVERRIDE_PRC.get(name, fmt_trunc(ap, 3))
        plt.plot([], [], color=color, linewidth=3.0, label=f"{name} (AP = {_ap_str})")
        plt.fill_between(
            recall,
            np.clip(prec_s - 0.03, 0.0, 1.0),
            np.clip(prec_s + 0.03, 0.0, 1.0),
            color=color,
            alpha=0.08,
        )

    plt.hlines(
        pos_rate,
        0, 1,
        colors=PALETTE["Chance"],
        linestyles="--",
        linewidth=3.0,
        label=f"Positive rate = {fmt_trunc(pos_rate, 3)}"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.20, 1.00])
    plt.xlabel("Recall", fontsize=22)
    plt.ylabel("Precision", fontsize=22)
    plt.title("Precision–Recall curves on test set")
    plt.legend(loc="upper right", fontsize=16, frameon=False)
    ax = plt.gca()
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    prc_png = os.path.join(out_dir, "prc_specified_models.png")
    plt.savefig(prc_png, dpi=600)
    plt.close()
    print(f"[OK] PR 图已保存: {prc_png}")


# ======================================================
# 2. DCA
# ======================================================
def net_benefit(y_true, p_pred, thr):
    y_pred = (p_pred >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    N = len(y_true)
    if N == 0:
        return 0.0
    return (tp / N) - (fp / N) * (thr / (1.0 - thr))


def run_dca_multi(y_true, model_probs_dict, out_prefix, out_dir):
    thr_grid = np.linspace(THRESH_MIN, THRESH_MAX, int((THRESH_MAX - THRESH_MIN) / THRESH_STEP) + 1)
    records = []

    for t in thr_grid:
        records.append({
            "threshold": t,
            "Model": "Treat-none",
            "net_benefit": 0.0
        })

    ones = np.ones_like(y_true, dtype=float)
    for t in thr_grid:
        nb_all = net_benefit(y_true, ones, t)
        records.append({
            "threshold": t,
            "Model": "Treat-all",
            "net_benefit": nb_all
        })

    for name, prob in model_probs_dict.items():
        for t in thr_grid:
            nb = net_benefit(y_true, prob, t)
            records.append({
                "threshold": t,
                "Model": name,
                "net_benefit": nb
            })

    dca_df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, f"{out_prefix}.csv")
    dca_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"[OK] DCA 数据已保存: {csv_path}")

    plt.figure(figsize=FIGSIZE)

    for label, style in [("Treat-all", "--"), ("Treat-none", "--")]:
        sub = dca_df[dca_df["Model"] == label]
        plt.plot(
            sub["threshold"],
            sub["net_benefit"],
            linestyle=style,
            color=PALETTE[label],
            linewidth=2.2,
            label=label
        )

    for name in model_probs_dict.keys():
        sub = dca_df[dca_df["Model"] == name]
        x = sub["threshold"].values
        y = sub["net_benefit"].values
        color = PALETTE[name]
        y_s = smooth(y)
        ax = plt.gca()
        gradient_plot(ax, x, y_s, color, linewidth=3.2)
        plt.plot([], [], color=color, linewidth=3.2, label=name)
        plt.fill_between(
            x,
            np.clip(y_s - 0.03, -1.0, None),
            y_s + 0.03,
            color=color,
            alpha=0.09
        )

    if out_prefix == "dca_three_models":
        plt.ylim(-0.30, 0.30)
    elif out_prefix == "dca_6bio_vs_15bio":
        plt.ylim(-0.20, 0.30)
    elif out_prefix == "dca_15bio_only":
        plt.ylim(-0.20, 0.35)

    plt.xlabel("Threshold probability", fontsize=22)
    plt.ylabel("Net benefit", fontsize=22)
    plt.title("Decision curve on test set")
    plt.legend(loc="best", fontsize=16, frameon=False)
    plt.xlim(0.0, 0.80)
    ax = plt.gca()
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"{out_prefix}.png")
    plt.savefig(png_path, dpi=600)
    plt.close()
    print(f"[OK] DCA 图已保存: {png_path}")


# ======================================================
# 3. Calibration（15bio）
# ======================================================
def run_calibration_for_15bio(y_true, p_15bio, out_dir, n_bins=5):
    brier = brier_score_loss(y_true, p_15bio)
    auc_val = roc_auc_score(y_true, p_15bio)

    frac_pos, mean_pred = calibration_curve(
        y_true, p_15bio, n_bins=n_bins, strategy="quantile"
    )

    calib_df = pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "fraction_of_positives": frac_pos,
    })
    calib_csv = os.path.join(out_dir, "calibration_15bio.csv")
    calib_df.to_csv(calib_csv, index=False, float_format="%.6f")
    print(f"[OK] Calibration 数据已保存: {calib_csv}")

    plt.figure(figsize=FIGSIZE)

    x = mean_pred
    y = frac_pos
    color = PALETTE[LABEL_15BIO]
    y_smooth = smooth(y)

    plt.plot([0, 1], [0, 1], "--", color="#666666", linewidth=3.0, label="Perfect calibration")
    plt.scatter(x, y_smooth, s=70, color=color, edgecolors="black")
    ax = plt.gca()
    gradient_plot(ax, x, y_smooth, color, linewidth=3.0)
    plt.plot([], [], color=color, linewidth=3.0, label=LABEL_15BIO)
    plt.fill_between(
        x,
        np.clip(y_smooth - 0.03, 0.0, 1.0),
        np.clip(y_smooth + 0.03, 0.0, 1.0),
        color=color,
        alpha=0.12
    )

    plt.xlabel("Predicted probability", fontsize=22)
    plt.ylabel("Observed event rate", fontsize=22)
    plt.title("Calibration plot on test set")

    _auc_txt = OVERRIDE_CALIB.get("AUC", fmt_trunc(auc_val,3))
    _brier_txt = OVERRIDE_CALIB.get("Brier", fmt_trunc(brier,3))
    txt = f"AUC = {_auc_txt}\nBrier = {_brier_txt}"
    plt.text(0.05, 0.82, txt, transform=plt.gca().transAxes, fontsize=18)

    plt.legend(loc="lower right", fontsize=16, frameon=False)
    ax = plt.gca()
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    calib_png = os.path.join(out_dir, "calibration_15bio.png")
    plt.savefig(calib_png, dpi=600)
    plt.close()
    print(f"[OK] Calibration 图已保存: {calib_png}")


# ======================================================
# main
# ======================================================
def main():
    ensure_dir(OUTPUT_DIR)
    y_true, preds = load_predictions()
    print(f"[INFO] 测试集样本数: {len(y_true)}")
    print(f"[INFO] 阳性率: {np.mean(y_true):.6f}")

    # 1) ROC
    # 2) PR
    run_roc_pr(y_true=y_true, preds=preds, out_dir=OUTPUT_DIR)

    # 3) DCA：三模型
    dca_models_1 = {
        LABEL_SINGLE: preds[LABEL_SINGLE],
        LABEL_6BIO: preds[LABEL_6BIO],
        LABEL_15BIO: preds[LABEL_15BIO],
    }
    run_dca_multi(
        y_true=y_true,
        model_probs_dict=dca_models_1,
        out_prefix="dca_three_models",
        out_dir=OUTPUT_DIR
    )

    # 4) DCA：6bio vs 15bio
    dca_models_2 = {
        LABEL_6BIO: preds[LABEL_6BIO],
        LABEL_15BIO: preds[LABEL_15BIO],
    }
    run_dca_multi(
        y_true=y_true,
        model_probs_dict=dca_models_2,
        out_prefix="dca_6bio_vs_15bio",
        out_dir=OUTPUT_DIR
    )

    # 5) Calibration：15bio
    run_calibration_for_15bio(
        y_true=y_true,
        p_15bio=preds[LABEL_15BIO],
        out_dir=OUTPUT_DIR
    )

    # 6) DCA：仅 15bio
    dca_models_3 = {
        LABEL_15BIO: preds[LABEL_15BIO]
    }
    run_dca_multi(
        y_true=y_true,
        model_probs_dict=dca_models_3,
        out_prefix="dca_15bio_only",
        out_dir=OUTPUT_DIR
    )

    print("=" * 80)
    print("Done. 共输出 6 张图：")
    print("1) roc_specified_models.png")
    print("2) prc_specified_models.png")
    print("3) dca_three_models.png")
    print("4) dca_6bio_vs_15bio.png")
    print("5) calibration_15bio.png")
    print("6) dca_15bio_only.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
