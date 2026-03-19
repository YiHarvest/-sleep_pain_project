# -*- coding: utf-8 -*-
"""
对最终 stacking 集成模型（Ensemble: logit_convex+T, GB+RF）
在测试集上做 Bootstrap 评估，得到 ROC AUC / PR AUC / Brier 等指标的
分布和 95% CI，并画一些直方图。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
sns.set_theme(style="whitegrid", context="talk", rc={"font.family": "Times New Roman"})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'

metric_color_map = {
    "ROC_AUC": "#8FBBD9",
    "PR_AUC": "#91D0A5",
    "Accuracy": "#E4D28A",
    "Sensitivity": "#9FD4E0",
    "Specificity": "#F7B7A3",
    "F1": "#E08C83",
    "Brier": "#aed9f5",
}

# ===================== 路径配置（按你本地情况检查/修改） =====================

# 数据路径：和 stacking_competition.py 中的 DATA_PATH 保持一致
DATA_PATH = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv"

# stacking 脚本输出目录：与 stacking_competition.py 的 OUT_DIR 完全一致
OUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\only_6_stacking_model_comparsion"

# 最终集成模型路径（stacking_competition.py 保存的模型）
MODEL_PATH = os.path.join(OUT_DIR, "final_ensemble_convex_T.pkl")

# 模型比较结果（读取 Thr_F1 / Thr_Youden）
SUMMARY_CSV = os.path.join(OUT_DIR, "stacking_model_comparison_summary.csv")

# 本脚本的 Bootstrap 输出目录：脚本同级目录下的 bootstrap_reslut
SCRIPT_DIR = Path(__file__).resolve().parent
BOOT_DIR = SCRIPT_DIR / "outputs/bootstrap_reslut"
BOOT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.RandomState(SEED)

# ============== 兼容：为反序列化提供两种集成类定义 ==============

class RankAveragePlattEnsemble:
    def __init__(self, preprocess, svm_model, cat_model, platt_lr):
        self.preprocess = preprocess
        self.svm_model = svm_model
        self.cat_model = cat_model
        self.platt_lr = platt_lr
    def predict_proba(self, X):
        p_svm = self.svm_model.predict_proba(X)[:, 1]
        p_cat = self.cat_model.predict_proba(X)[:, 1]
        n = len(p_svm)
        ranks = []
        for p in [p_svm, p_cat]:
            order = np.argsort(np.argsort(p))
            r = (order + 1) / n
            ranks.append(r)
        r_mean = np.mean(ranks, axis=0).reshape(-1, 1)
        return self.platt_lr.predict_proba(r_mean)

class LogitConvexTEnsemble:
    def __init__(self, preprocess, svm_model, cat_model, w_opt, T_opt):
        self.preprocess = preprocess
        self.svm_model = svm_model
        self.cat_model = cat_model
        self.w_opt = np.array(w_opt)
        self.T_opt = float(T_opt)
    def _safe_logit(self, p):
        p = np.clip(p, 1e-7, 1-1e-7)
        return np.log(p / (1 - p))
    def _expit(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    def predict_proba(self, X):
        p_svm = self.svm_model.predict_proba(X)[:, 1]
        p_cat = self.cat_model.predict_proba(X)[:, 1]
        z = self.w_opt[0] * self._safe_logit(p_svm) + self.w_opt[1] * self._safe_logit(p_cat)
        p1 = self._expit(z / (self.T_opt + 1e-12))
        return np.column_stack([1.0 - p1, p1])

# 尝试将 stacking_competition 中的同名类映射到 __main__，以兼容不同的持久化上下文
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import stacking_competition as _sc
    setattr(_sys.modules[__name__], 'RankAveragePlattEnsemble', getattr(_sc, 'RankAveragePlattEnsemble', RankAveragePlattEnsemble))
    setattr(_sys.modules[__name__], 'LogitConvexTEnsemble', getattr(_sc, 'LogitConvexTEnsemble', LogitConvexTEnsemble))
except Exception:
    pass

# ===================== 一些小工具函数 =====================

def auto_detect_target(df: pd.DataFrame) -> str:
    """自动识别二分类目标列。"""
    candidate = ["Chronic_pain", "Pain", "pain", "Outcome", "outcome",
                 "label", "Label", "target", "Target", "y", "Y"]
    for c in candidate:
        if c in df.columns:
            v = df[c].dropna().unique()
            if set(np.unique(v)).issubset({0, 1}) and len(np.unique(v)) == 2:
                return c
    last = df.columns[-1]
    v = df[last].dropna().unique()
    if set(np.unique(v)).issubset({0, 1}) and len(np.unique(v)) == 2:
        return last
    raise ValueError("未能自动识别到二分类目标列，请手动指定。")


def compute_metrics(y_true, y_prob, threshold):
    """与 stacking_competition.py 中一致的二分类指标计算函数。"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    youden = recall + specificity - 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        "Accuracy": acc,
        "Precision/PPV": precision,
        "Recall/Sensitivity": recall,
        "Specificity": specificity,
        "F1": f1,
        "NPV": npv,
        "Youden": youden
    }


def bootstrap_metrics(y_true, y_prob, thr_f1, thr_yj,
                      n_bootstrap=1000, random_state=42):
    """在固定测试集上对给定概率做 bootstrap 评估。"""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    idx = np.arange(n)

    records = []

    for b in range(n_bootstrap):
        sample_idx = rng.choice(idx, size=n, replace=True)
        yt = y_true[sample_idx]
        yp = y_prob[sample_idx]

        # 有时抽样可能只包含单一类别，AUC 无法计算，跳过
        try:
            roc = roc_auc_score(yt, yp)
            pr = average_precision_score(yt, yp)
        except ValueError:
            continue

        brier = brier_score_loss(yt, yp)

        met_f1 = compute_metrics(yt, yp, thr_f1)
        met_yj = compute_metrics(yt, yp, thr_yj)

        row = {
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "Brier": brier,
        }
        # 在 F1 阈值下的指标
        for k, v in met_f1.items():
            row[f"F1thr_{k}"] = v
        # 在 Youden 阈值下的指标
        for k, v in met_yj.items():
            row[f"YoudenThr_{k}"] = v

        records.append(row)

    return pd.DataFrame(records)


def summarize_bootstrap(df: pd.DataFrame, cols):
    """对指定指标做均值/标准差/95% CI 汇总。"""
    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        mean = s.mean()
        std = s.std(ddof=1)
        lower = s.quantile(0.025)
        upper = s.quantile(0.975)
        rows.append({
            "Metric": c,
            "Mean": mean,
            "Std": std,
            "CI_lower_2.5%": lower,
            "CI_upper_97.5%": upper
        })
    return pd.DataFrame(rows)


# ===================== 主流程 =====================

def main():
    # 1. 读取固定测试集（不再随机划分）
    df = pd.read_csv(DATA_PATH)
    target_col = auto_detect_target(df)
    X_te_df = df.drop(columns=[target_col])
    y_te = df[target_col].values

    # 2. 加载最终集成模型（Convex+T）
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到最终模型文件：{MODEL_PATH}")
    try:
        ensemble = joblib.load(MODEL_PATH)
    except Exception:
        print("[WARN] 直接加载最终模型失败")
        raise

    # 3. 在测试集上得到预测概率（避免双重前处理，直接用基模型管线 + Platt）
    p_te = None
    try:
        p = ensemble.predict_proba(X_te_df)
        p_te = p[:, 1] if isinstance(p, np.ndarray) and p.ndim == 2 else np.array(p).ravel()
    except Exception:
        # 回退：若为 Rank+Platt 结构
        try:
            p_svm_te = ensemble.svm_model.predict_proba(X_te_df)[:, 1]
            p_cat_te = ensemble.cat_model.predict_proba(X_te_df)[:, 1]
            n_te = len(X_te_df)
            r1 = (np.argsort(np.argsort(p_svm_te)) + 1) / float(n_te)
            r2 = (np.argsort(np.argsort(p_cat_te)) + 1) / float(n_te)
            s_te = ((r1 + r2) / 2.0).reshape(-1, 1)
            p_te = ensemble.platt_lr.predict_proba(s_te)[:, 1]
        except Exception:
            raise

    # 4. 从 summary CSV 中读取该模型的阈值（Thr_F1, Thr_Youden）
    summary = pd.read_csv(SUMMARY_CSV)
    row = summary.loc[summary["Model"] == "Ensemble: logit_convex+T (GB+RF)"]
    if row.empty:
        raise ValueError("在 stacking_model_comparison_summary.csv 中未找到目标模型行。")
    thr_f1 = float(row["Thr_F1"].iloc[0])
    thr_yj = float(row["Thr_Youden"].iloc[0])
    print(f"[INFO] 使用阈值：Thr_F1 = {thr_f1:.3f}, Thr_Youden = {thr_yj:.3f}")

    # 5. 先计算“原始测试集”的单次指标，方便对比
    base_metrics = {
        "test_ROC_AUC": roc_auc_score(y_te, p_te),
        "test_PR_AUC": average_precision_score(y_te, p_te),
        "test_Brier": brier_score_loss(y_te, p_te),
    }
    base_metrics.update({f"F1thr_{k}": v for k, v in compute_metrics(y_te, p_te, thr_f1).items()})
    base_metrics.update({f"YoudenThr_{k}": v for k, v in compute_metrics(y_te, p_te, thr_yj).items()})
    base_metrics_df = pd.DataFrame([base_metrics])
    base_metrics_df.to_csv(BOOT_DIR / "test_metrics_convex_T.csv",
                           index=False, encoding="utf-8-sig")

    # 6. 做 bootstrap
    print("[INFO] 开始 bootstrap 评估 ...")
    boot_df = bootstrap_metrics(y_te, p_te, thr_f1, thr_yj,
                                n_bootstrap=1000, random_state=SEED)
    boot_csv = BOOT_DIR / "bootstrap_metrics_convex_T.csv"
    boot_df.to_csv(boot_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存每次 bootstrap 结果 → {boot_csv}")

    # 7. 汇总为均值 + 标准差 + 95% CI
    cols_to_sum = [
        "ROC_AUC", "PR_AUC", "Brier",
        "F1thr_F1", "F1thr_Accuracy", "F1thr_Precision/PPV", "F1thr_Recall/Sensitivity", "F1thr_Specificity", "F1thr_NPV",
        "YoudenThr_F1", "YoudenThr_Accuracy", "YoudenThr_Precision/PPV", "YoudenThr_Recall/Sensitivity", "YoudenThr_Specificity", "YoudenThr_NPV"
    ]
    summary_df = summarize_bootstrap(boot_df, cols_to_sum)
    summary_csv = BOOT_DIR / "bootstrap_summary_convex_T.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已保存 bootstrap 汇总结果 → {summary_csv}")

    # 8. 画几个直方图（AUC / PR-AUC / F1）
    plt.figure(figsize=(10,6))
    data_roc = boot_df["ROC_AUC"].dropna().values
    counts_roc, bins_roc, _ = plt.hist(
        data_roc, bins=30, color=metric_color_map["ROC_AUC"], edgecolor="black", linewidth=0.75
    )
    plt.axvline(base_metrics["test_ROC_AUC"], color="black", linestyle="--", linewidth=2, label="Single-test metric")
    row_roc = summary_df.loc[summary_df["Metric"] == "ROC_AUC"]
    if not row_roc.empty:
        mu = float(row_roc["Mean"].iloc[0])
        sigma = float(row_roc["Std"].iloc[0])
        x = np.linspace(bins_roc[0], bins_roc[-1], 200)
        bw = bins_roc[1] - bins_roc[0]
        y = norm.pdf(x, mu, sigma) * len(data_roc) * bw
        plt.plot(x, y, color="red", linewidth=2, label="Normal fit curve")
    plt.title("Bootstrap AUC (Convex+T)", fontsize=36, fontweight='bold')
    plt.xlabel("AUC", fontsize=28)
    plt.ylabel("Frequency", fontsize=28)
    plt.legend(loc="upper left")
    ax = plt.gca()
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    plt.xticks(rotation=0, fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()
    plt.savefig(BOOT_DIR / "hist_roc_auc_convex_T.png", dpi=600)
    plt.savefig(BOOT_DIR / "hist_roc_auc_convex_T.tif", dpi=600)
    plt.close()

    plt.figure(figsize=(10,6))
    data_pr = boot_df["PR_AUC"].dropna().values
    counts_pr, bins_pr, _ = plt.hist(
        data_pr, bins=30, color=metric_color_map["PR_AUC"], edgecolor="black", linewidth=0.75
    )
    plt.axvline(base_metrics["test_PR_AUC"], color="black", linestyle="--", linewidth=2, label="Single-test metric")
    row_pr = summary_df.loc[summary_df["Metric"] == "PR_AUC"]
    if not row_pr.empty:
        mu = float(row_pr["Mean"].iloc[0])
        sigma = float(row_pr["Std"].iloc[0])
        x = np.linspace(bins_pr[0], bins_pr[-1], 200)
        bw = bins_pr[1] - bins_pr[0]
        y = norm.pdf(x, mu, sigma) * len(data_pr) * bw
        plt.plot(x, y, color="red", linewidth=2, label="Normal fit curve")
    plt.title("Bootstrap AP (Convex+T)", fontsize=36, fontweight='bold')
    plt.xlabel("AP", fontsize=28)
    plt.ylabel("Frequency", fontsize=28)
    plt.legend(loc="upper left")
    ax = plt.gca()
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    plt.xticks(rotation=0, fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()
    plt.savefig(BOOT_DIR / "hist_pr_auc_convex_T.png", dpi=600)
    plt.savefig(BOOT_DIR / "hist_pr_auc_convex_T.tif", dpi=600)
    plt.close()

    if "F1thr_F1" in boot_df.columns:
        plt.figure(figsize=(10,6))
        data_f1 = boot_df["F1thr_F1"].dropna().values
        counts_f1, bins_f1, _ = plt.hist(
            data_f1, bins=30, color=metric_color_map["F1"], edgecolor="black", linewidth=0.75
        )
        plt.axvline(base_metrics["F1thr_F1"], color="black", linestyle="--", linewidth=2, label="Single-test metric")
        row_f1 = summary_df.loc[summary_df["Metric"] == "F1thr_F1"]
        if not row_f1.empty:
            mu = float(row_f1["Mean"].iloc[0])
            sigma = float(row_f1["Std"].iloc[0])
            x = np.linspace(bins_f1[0], bins_f1[-1], 200)
            bw = bins_f1[1] - bins_f1[0]
            y = norm.pdf(x, mu, sigma) * len(data_f1) * bw
            plt.plot(x, y, color="red", linewidth=2, label="Normal fit curve")
        plt.title("Bootstrap F1(Convex+T)", fontsize=36, fontweight='bold')
        plt.xlabel("F1", fontsize=28)
        plt.ylabel("Frequency", fontsize=28)
        plt.legend(loc="upper left")
        ax = plt.gca()
        ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
        plt.xticks(rotation=0, fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        plt.savefig(BOOT_DIR / "hist_f1_f1thr_convex_T.png", dpi=600)
        plt.savefig(BOOT_DIR / "hist_f1_f1thr_convex_T.tif", dpi=600)
        plt.close()

    if "F1thr_Accuracy" in boot_df.columns:
        plt.figure(figsize=(10,6))
        data_acc = boot_df["F1thr_Accuracy"].dropna().values
        counts_acc, bins_acc, _ = plt.hist(
            data_acc, bins=30, color=metric_color_map["Accuracy"], edgecolor="black", linewidth=0.75
        )
        plt.axvline(base_metrics["F1thr_Accuracy"], color="black", linestyle="--", linewidth=2, label="Single-test metric")
        row_acc = summary_df.loc[summary_df["Metric"] == "F1thr_Accuracy"]
        if not row_acc.empty:
            mu = float(row_acc["Mean"].iloc[0])
            sigma = float(row_acc["Std"].iloc[0])
            x = np.linspace(bins_acc[0], bins_acc[-1], 200)
            bw = bins_acc[1] - bins_acc[0]
            y = norm.pdf(x, mu, sigma) * len(data_acc) * bw
            plt.plot(x, y, color="red", linewidth=2, label="Normal fit curve")
        plt.title("Bootstrap Accuracy at F1-opt threshold (Convex+T)", fontsize=36, fontweight='bold')
        plt.xlabel("Accuracy", fontsize=28)
        plt.ylabel("Frequency", fontsize=28)
        plt.legend(loc="upper left")
        ax = plt.gca()
        ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
        plt.xticks(rotation=0, fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        plt.savefig(BOOT_DIR / "hist_accuracy_f1thr_convex_T.png", dpi=600)
        plt.savefig(BOOT_DIR / "hist_accuracy_f1thr_convex_T.tif", dpi=600)
        plt.close()

    if "F1thr_Recall/Sensitivity" in boot_df.columns:
        plt.figure(figsize=(10,6))
        data_sens = boot_df["F1thr_Recall/Sensitivity"].dropna().values
        counts_sens, bins_sens, _ = plt.hist(
            data_sens, bins=30, color=metric_color_map["Sensitivity"], edgecolor="black", linewidth=0.75
        )
        plt.axvline(base_metrics["F1thr_Recall/Sensitivity"], color="black", linestyle="--", linewidth=2, label="Single-test metric")
        row_sens = summary_df.loc[summary_df["Metric"] == "F1thr_Recall/Sensitivity"]
        if not row_sens.empty:
            mu = float(row_sens["Mean"].iloc[0])
            sigma = float(row_sens["Std"].iloc[0])
            x = np.linspace(bins_sens[0], bins_sens[-1], 200)
            bw = bins_sens[1] - bins_sens[0]
            y = norm.pdf(x, mu, sigma) * len(data_sens) * bw
            plt.plot(x, y, color="red", linewidth=2, label="Normal fit curve")
        plt.title("Bootstrap Sensitivity at F1-opt threshold (Convex+T)", fontsize=36, fontweight='bold')
        plt.xlabel("Sensitivity", fontsize=28)
        plt.ylabel("Frequency", fontsize=28)
        plt.legend(loc="upper left")
        ax = plt.gca()
        ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
        plt.xticks(rotation=0, fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        plt.savefig(BOOT_DIR / "hist_sensitivity_f1thr_convex_T.png", dpi=600)
        plt.savefig(BOOT_DIR / "hist_sensitivity_f1thr_convex_T.tif", dpi=600)
        plt.close()

    if "F1thr_Specificity" in boot_df.columns:
        plt.figure(figsize=(10,6))
        data_spec = boot_df["F1thr_Specificity"].dropna().values
        counts_spec, bins_spec, _ = plt.hist(
            data_spec, bins=30, color=metric_color_map["Specificity"], edgecolor="black", linewidth=0.75
        )
        plt.axvline(base_metrics["F1thr_Specificity"], color="black", linestyle="--", linewidth=2, label="Single-test metric")
        row_spec = summary_df.loc[summary_df["Metric"] == "F1thr_Specificity"]
        if not row_spec.empty:
            mu = float(row_spec["Mean"].iloc[0])
            sigma = float(row_spec["Std"].iloc[0])
            x = np.linspace(bins_spec[0], bins_spec[-1], 200)
            bw = bins_spec[1] - bins_spec[0]
            y = norm.pdf(x, mu, sigma) * len(data_spec) * bw
            plt.plot(x, y, color="red", linewidth=2, label="Normal fit curve")
        plt.title("Bootstrap Specificity at F1-opt threshold (Convex+T)", fontsize=36, fontweight='bold')
        plt.xlabel("Specificity", fontsize=28)
        plt.ylabel("Frequency", fontsize=28)
        plt.legend(loc="upper left")
        ax = plt.gca()
        ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
        plt.xticks(rotation=0, fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        plt.savefig(BOOT_DIR / "hist_specificity_f1thr_convex_T.png", dpi=600)
        plt.savefig(BOOT_DIR / "hist_specificity_f1thr_convex_T.tif", dpi=600)
        plt.close()

    if "Brier" in boot_df.columns:
        plt.figure(figsize=(10,6))
        data_brier = boot_df["Brier"].dropna().values
        counts_brier, bins_brier, _ = plt.hist(
            data_brier, bins=30, color=metric_color_map["Brier"], edgecolor="black", linewidth=0.75
        )
        plt.axvline(base_metrics["test_Brier"], color="black", linestyle="--", linewidth=2, label="Single-test metric")
        row_brier = summary_df.loc[summary_df["Metric"] == "Brier"]
        if not row_brier.empty:
            mu = float(row_brier["Mean"].iloc[0])
            sigma = float(row_brier["Std"].iloc[0])
            x = np.linspace(bins_brier[0], bins_brier[-1], 200)
            bw = bins_brier[1] - bins_brier[0]
            y = norm.pdf(x, mu, sigma) * len(data_brier) * bw
            plt.plot(x, y, color="red", linewidth=2, label="Normal fit curve")
        plt.title("Bootstrap Brier (Convex+T)", fontsize=36, fontweight='bold')
        plt.xlabel("Brier", fontsize=28)
        plt.ylabel("Frequency", fontsize=28)
        plt.legend(loc="upper left")
        ax = plt.gca()
        ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
        plt.xticks(rotation=0, fontsize=28)
        plt.yticks(fontsize=28)
        plt.tight_layout()
        plt.savefig(BOOT_DIR / "hist_brier_convex_T.png", dpi=600)
        plt.savefig(BOOT_DIR / "hist_brier_convex_T.tif", dpi=600)
        plt.close()

        print("[DONE] Bootstrap 评估与图像绘制完成。")


if __name__ == "__main__":
    main()
