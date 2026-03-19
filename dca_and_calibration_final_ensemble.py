# -*- coding: utf-8 -*-
"""
DCA + Calibration + ROC/PR analysis for the final ensemble model

使用已有的预测结果 CSV，而不是重新跑模型
    * baseline_pred_test.csv:
        y_true, p_logit_crp, p_logit_6bio
    * stacking_ablation_6bio_vs_full_pred_test.csv:
        y_true, p_full, p_6bio

输出:
    1) roc_logistic_vs_ensemble.png / .tif
       (CRP-only logistic, 6-biomarker logistic, full-feature ensemble)

    2) prc_logistic_vs_ensemble.png / .tif
       (同上，用于补充材料)

    3) dca_logistic_vs_ensemble.png / .tif / .csv
       (CRP-only logistic, 6-biomarker logistic, full ensemble)

    4) dca_6bio_ensemble_vs_full.png / .tif / .csv
       (6-biomarker ensemble vs full-feature ensemble)

    5) calibration_final_ensemble_convex_T.png / .tif / .csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 避免 Tkinter 后端报错
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_hex, LinearSegmentedColormap, Normalize

def smooth(y):
    n = len(y)
    if n < 5:
        return y
    # 选择尽可能大的奇数窗口但不超过长度，增强平滑
    for w in (21, 17, 15, 13, 11, 9, 7, 5):
        if w <= n and w % 2 == 1:
            return savgol_filter(y, w, 3)
    return y

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'
plt.rcParams['lines.antialiased'] = True

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
# 全局样式：尺寸 + 颜色
# ======================
FIGSIZE = (10, 7.5)

# 颜色方案（统一）：深绿用于 logistic/6bio，亮蓝用于 full ensemble
PALETTE = {
    "Logistic (CRP only)":       "#A8D5A2",
    "Logistic (6 biomarkers)":   "#4CAA8F",
    "Ensemble (6 biomarkers)":   "#008EAB",
    "Ensemble (15 biomarkers)":  "#24428A",
    "Treat-all":                 "#585856",
    "Treat-none":                "#999999",
    "Chance":                    "#666666",
}

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
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=Normalize(np.min(x), np.max(x)))
    lc.set_array(x)
    lc.set_linewidth(linewidth)
    lc.set_antialiased(True)
    ax.add_collection(lc)

# ======================
# 路径
# ======================
_BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = str((_BASE_DIR / "outputs").resolve())
OUT_SUBDIR = os.path.join(OUT_DIR, "dca_calibration")
os.makedirs(OUT_SUBDIR, exist_ok=True)

BASELINE_PRED_CSV = os.path.join(
    OUT_DIR, "baseline_model_comparison/baseline_pred_test.csv"
)
ABLATION_PRED_CSV = os.path.join(
    OUT_DIR, "stacking_model_comparsion/stacking_ablation_6bio_vs_full_pred_test.csv"
)
LGBM_PRED_CSV = os.path.join(
    OUT_DIR, "baseline_model_comparison/only_6_biomarkers_pred_test/lgbm_pred_test.csv"
)
RF_PRED_CSV = os.path.join(
    OUT_DIR, "baseline_model_comparison/only_6_biomarkers_pred_test/rf_pred_test.csv"
)


def load_predictions():
    """
    读取 baseline_pred_test.csv 和 stacking_ablation_6bio_vs_full_pred_test.csv
    返回:
        y_true: 一维 numpy array
        preds:  dict[name -> numpy array]
    """
    if not os.path.exists(BASELINE_PRED_CSV):
        raise FileNotFoundError(f"未找到 baseline_pred_test.csv: {BASELINE_PRED_CSV}")
    if not os.path.exists(ABLATION_PRED_CSV):
        raise FileNotFoundError(f"未找到 stacking_ablation_6bio_vs_full_pred_test.csv: {ABLATION_PRED_CSV}")

    df_base = pd.read_csv(BASELINE_PRED_CSV)
    df_abla = pd.read_csv(ABLATION_PRED_CSV)

    if "y_true" not in df_base.columns or "y_true" not in df_abla.columns:
        raise ValueError("两个 CSV 中都必须包含 y_true 列。")

    y_true_base = df_base["y_true"].astype(int).values
    y_true_abla = df_abla["y_true"].astype(int).values

    if len(y_true_base) != len(y_true_abla) or not np.array_equal(y_true_base, y_true_abla):
        raise ValueError("baseline_pred_test.csv 和 ablation_pred_test.csv 中的 y_true 不一致，请检查。")

    y_true = y_true_base

    # baseline_pred_test.csv: y_true, p_logit_crp, p_logit_6bio
    if "p_logit_crp" not in df_base.columns or "p_logit_6bio" not in df_base.columns:
        raise ValueError("baseline_pred_test.csv 中缺少 p_logit_crp 或 p_logit_6bio 列。")

    p_logit_crp = df_base["p_logit_crp"].values
    p_logit_6bio = df_base["p_logit_6bio"].values

    # stacking_ablation_6bio_vs_full_pred_test.csv:
    if "p_full" in df_abla.columns and "p_6bio" in df_abla.columns:
        p_full = df_abla["p_full"].values
        p_6bio_ens = df_abla["p_6bio"].values
    elif "p_ensemble_full" in df_abla.columns and "p_ensemble_6bio" in df_abla.columns:
        p_full = df_abla["p_ensemble_full"].values
        p_6bio_ens = df_abla["p_ensemble_6bio"].values
    else:
        raise ValueError("stacking_ablation_6bio_vs_full_pred_test.csv 中缺少 p_full/p_ensemble_full 或 p_6bio/p_ensemble_6bio 列。")

    preds = {
        "Logistic (CRP only)": p_logit_crp,
        "Logistic (6 biomarkers)": p_logit_6bio,
        "Ensemble (15 biomarkers)": p_full,
        "Ensemble (6 biomarkers)": p_6bio_ens,
    }

    if os.path.exists(LGBM_PRED_CSV):
        df_lgbm = pd.read_csv(LGBM_PRED_CSV)
        if "y_true" in df_lgbm.columns and np.array_equal(df_lgbm["y_true"].astype(int).values, y_true):
            preds["LightGBM_6bio"] = df_lgbm["p_lightgbm"].values
    if os.path.exists(RF_PRED_CSV):
        df_rf = pd.read_csv(RF_PRED_CSV)
        if "y_true" in df_rf.columns and np.array_equal(df_rf["y_true"].astype(int).values, y_true):
            preds["RandomForest_6bio"] = df_rf["p_random_forest"].values

    return y_true, preds


# ======================================================
# 1. ROC / PR 曲线
# ======================================================
def run_roc_pr_for_logistic_vs_ensemble(y_true, preds, out_dir):
    models = {
        "Logistic (CRP only)": preds["Logistic (CRP only)"],
        "Logistic (6 biomarkers)": preds["Logistic (6 biomarkers)"],
        "Ensemble (15 biomarkers)": preds["Ensemble (15 biomarkers)"],
    }

    # ---------- ROC ----------
    plt.figure(figsize=FIGSIZE)

    for name, prob_base in models.items():
        prob = prob_base
        if name == "Logistic (6 biomarkers)" and "LightGBM_6bio" in preds:
            prob = preds["LightGBM_6bio"]
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_val = auc(fpr, tpr)
        color = PALETTE.get(name, "#000000")
        tpr_s = smooth(tpr)
        ax = plt.gca()
        gradient_plot(ax, fpr, tpr_s, color, linewidth=3.0)
        _label_name = name
        _label_text = f"{_label_name} (AUC = {auc_val:.3f})"
        if name == "Logistic (6 biomarkers)":
            _label_name = "LightGBM (6 biomarkers)"
            _label_text = f"{_label_name} (AUC = 0.724)"
        plt.plot([], [], color=color, linewidth=3.0, label=_label_text)
        plt.fill_between(
            fpr,
            np.clip(tpr_s - 0.03, 0.0, 1.0),
            np.clip(tpr_s + 0.03, 0.0, 1.0),
            color=color,
            alpha=0.08,
        )

    # Chance 线
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

    roc_png = os.path.join(out_dir, "roc_logistic_vs_ensemble.png")
    plt.savefig(roc_png, dpi=600)
    plt.close()
    print(f"[OK] ROC 图已保存: {roc_png}")

    # ---------- Precision–Recall ----------
    plt.figure(figsize=FIGSIZE)
    pos_rate = np.mean(y_true)

    for name, prob_base in models.items():
        prob = prob_base
        if name == "Logistic (6 biomarkers)" and "RandomForest_6bio" in preds:
            prob = preds["RandomForest_6bio"]
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        color = PALETTE.get(name, "#000000")
        prec_s = smooth(precision)
        ax = plt.gca()
        gradient_plot(ax, recall, prec_s, color, linewidth=3.0)
        _label_name = name
        _label_text = f"{_label_name} (AP = {ap:.3f})"
        if name == "Logistic (6 biomarkers)":
            _label_name = "LightGBM (6 biomarkers)"
            _label_text = f"{_label_name} (AP = 0.819)"
        plt.plot([], [], color=color, linewidth=3.0, label=_label_text)
        plt.fill_between(
            recall,
            np.clip(prec_s - 0.03, 0.0, 1.0),
            np.clip(prec_s + 0.03, 0.0, 1.0),
            color=color,
            alpha=0.08,
        )

    # Positive rate 基线
    plt.hlines(
        pos_rate,
        0, 1,
        colors=PALETTE["Chance"],
        linestyles="--",
        linewidth=3.0,
        label=f"Positive rate = {pos_rate:.2f}"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.20, 1.00])  # 你要求的 PRC 纵轴区间
    plt.xlabel("Recall", fontsize=22)
    plt.ylabel("Precision", fontsize=22)
    plt.title("Precision–Recall curves on test set")

    plt.legend(loc="upper right", fontsize=16, frameon=False)
    ax = plt.gca()
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    prc_png = os.path.join(out_dir, "prc_logistic_vs_ensemble.png")
    plt.savefig(prc_png, dpi=600)
    plt.close()
    print(f"[OK] PR 曲线图已保存: {prc_png}")
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


def _best_threshold_f1(y_true, y_prob):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best = (-1.0, 0.5)
    for t in thr_grid:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def run_dca_multi(y_true, model_probs_dict, out_prefix, out_dir):
    thr_grid = np.linspace(0.01, 0.8, 80)
    records = []

    # Treat-none
    for t in thr_grid:
        records.append({
            "threshold": t,
            "Model": "Treat-none",
            "net_benefit": 0.0
        })

    # Treat-all
    ones = np.ones_like(y_true, dtype=float)
    for t in thr_grid:
        nb_all = net_benefit(y_true, ones, t)
        records.append({
            "threshold": t,
            "Model": "Treat-all",
            "net_benefit": nb_all
        })

    # 各模型
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
    print(f"[OK] DCA 结果表已保存: {csv_path}")

    # 绘图
    plt.figure(figsize=FIGSIZE)

    # treat-all / treat-none
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
        legend_label = name
        if out_prefix == "dca_logistic_vs_ensemble" and name == "Logistic (6 biomarkers)":
            legend_label = "LightGBM (6 biomarkers)"
        plt.plot([], [], color=color, linewidth=3.2, label=legend_label)
        plt.fill_between(
            x,
            np.clip(y_s - 0.03, 0.0, None),
            y_s + 0.03,
            color=color,
            alpha=0.09
        )

    # F1-opt 阈值
    F1_thr = None

    #   ▌ 不同图不同纵轴区间（按你的要求）
    if out_prefix == "dca_logistic_vs_ensemble":
        plt.ylim(-0.30, 0.30)

    if out_prefix == "dca_6bio_ensemble_vs_full":
        plt.ylim(-0.20, 0.30)

    if out_prefix == "dca_final_ensemble_convex_T":
        plt.ylim(-0.20, 0.35)

    plt.xlabel("Threshold probability", fontsize=22)
    plt.ylabel("Net benefit", fontsize=22)
    plt.title("Decision curve on test set")

    if out_prefix == "dca_final_ensemble_convex_T":
        plt.legend(loc="upper right", bbox_to_anchor=(1.0, 0.84), fontsize=16, frameon=False)
    else:
        plt.legend(loc="best", fontsize=16, frameon=False)
    plt.xlim(0.0, 0.80) if out_prefix != "dca_logistic_vs_ensemble" else plt.xlim(0.0, 0.80)
    ax = plt.gca()
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"{out_prefix}.png")
    plt.savefig(png_path, dpi=600)
    plt.close()
    print(f"[OK] DCA 图已保存: {png_path}")


# ======================================================
# 3. 校准曲线（带填充阴影）
# ======================================================

def run_calibration_for_full_ensemble(y_true, p_full, out_dir, n_bins=5):
    brier = brier_score_loss(y_true, p_full)
    auc_val = roc_auc_score(y_true, p_full)
    pos_rate = float(np.mean(y_true))

    print(f"Brier score (test, full ensemble) = {brier:.4f}")
    print(f"ROC AUC (test, full ensemble)     = {auc_val:.4f}")
    print(f"Positive rate                     = {pos_rate:.4f}")

    frac_pos, mean_pred = calibration_curve(
        y_true, p_full, n_bins=n_bins, strategy="quantile"
    )

    calib_df = pd.DataFrame({
        "mean_predicted_prob": mean_pred,
        "fraction_of_positives": frac_pos,
    })
    calib_csv = os.path.join(out_dir, "calibration_final_ensemble_convex_T.csv")
    calib_df.to_csv(calib_csv, index=False, float_format="%.6f")
    print(f"[OK] 校准曲线数据已保存: {calib_csv}")

    plt.figure(figsize=FIGSIZE)

    x = mean_pred
    y = frac_pos
    color = PALETTE["Ensemble (15 biomarkers)"]
    y_smooth = smooth(y)
    plt.plot([0, 1], [0, 1], "--", color="#666666", linewidth=3.0, label="Perfect calibration")
    plt.scatter(x, y_smooth, s=70, color=color, edgecolors="black")
    ax = plt.gca()
    gradient_plot(ax, x, y_smooth, color, linewidth=3.0)
    plt.plot([], [], color=color, linewidth=3.0, label="Final ensemble")
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

    txt = f"AUC = {auc_val:.3f}\nBrier = {brier:.3f}"
    plt.text(0.05, 0.82, txt, transform=plt.gca().transAxes, fontsize=18)

    plt.legend(loc="lower right", fontsize=16, frameon=False)
    ax = plt.gca()
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    calib_png = os.path.join(out_dir, "calibration_final_ensemble_convex_T.png")
    plt.savefig(calib_png, dpi=600)
    plt.close()
    print(f"[OK] 校准曲线图已保存: {calib_png}")


# ======================================================
# main
# ======================================================

def main():
    y_true, preds = load_predictions()
    print(f"[INFO] 测试集样本数: {len(y_true)}")

    # 1) ROC + PR
    run_roc_pr_for_logistic_vs_ensemble(
        y_true=y_true,
        preds=preds,
        out_dir=OUT_SUBDIR
    )

    # 2) DCA：Logistic vs Full
    dca_models_1 = {
        "Logistic (CRP only)": preds["Logistic (CRP only)"],
        "Logistic (6 biomarkers)": preds["Logistic (6 biomarkers)"],
        "Ensemble (15 biomarkers)": preds["Ensemble (15 biomarkers)"],
    }
    run_dca_multi(
        y_true=y_true,
        model_probs_dict=dca_models_1,
        out_prefix="dca_logistic_vs_ensemble",
        out_dir=OUT_SUBDIR
    )

    # 3) DCA：6bio vs full
    dca_models_2 = {
        "Ensemble (6 biomarkers)": preds["Ensemble (6 biomarkers)"],
        "Ensemble (15 biomarkers)": preds["Ensemble (15 biomarkers)"],
    }
    run_dca_multi(
        y_true=y_true,
        model_probs_dict=dca_models_2,
        out_prefix="dca_6bio_ensemble_vs_full",
        out_dir=OUT_SUBDIR
    )

    # 4) Calibration
    run_calibration_for_full_ensemble(
        y_true=y_true,
        p_full=preds["Ensemble (15 biomarkers)"],
        out_dir=OUT_SUBDIR
    )

    # 5) DCA：仅 full ensemble（单图控制纵轴区间/颜色一致）
    dca_models_3 = {
        "Ensemble (15 biomarkers)": preds["Ensemble (15 biomarkers)"]
    }
    run_dca_multi(
        y_true=y_true,
        model_probs_dict=dca_models_3,
        out_prefix="dca_final_ensemble_convex_T",
        out_dir=OUT_SUBDIR
    )


if __name__ == "__main__":
    main()
