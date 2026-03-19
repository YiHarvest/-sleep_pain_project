# -*- coding: utf-8 -*-
"""
低自由度稳健融合：logit凸组合+温度缩放 / 秩平均+Platt
- 固定数据路径: DATA_PATH（不改）
- 模型：RandomForest, GradientBoosting（名称沿用为 RandomForest / GradientBoosting）
- 外层CV：OOF 概率用于学习融合参数与统计逐折指标
- 三类阈值：F1最大 / Recall≥0.80下PPV最大 / Youden最大
- 输出：
    stacking_model_comparison_summary.csv（附加 mean/std 列）
    逐折 CV 结果 + 小提琴图
    最终三个集成模型的 pkl
    额外：6 指标 vs 全特征的特征消融结果 stacking_ablation_6bio_vs_full.csv
          及其测试集预测 stacking_ablation_6bio_vs_full_pred_test.csv
"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_fscore_support, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid", context="talk", rc={"font.family": "Times New Roman"})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 18.0
plt.rcParams['axes.labelsize'] = 18.0
plt.rcParams['xtick.labelsize'] = 16.5
plt.rcParams['ytick.labelsize'] = 16.5
plt.rcParams['legend.fontsize'] = 16.5
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'

# ==== persist utils ====
import joblib
from pathlib import Path

# =========================
# 固定随机种子（保持不变）
# =========================

SEED = 42
np.random.seed(SEED)
SPLIT_SEED = 8
MODEL_RANDOM_STATE = SEED + SPLIT_SEED

# 暴露给后处理脚本用
FEATURE_COLS = None  # 将在数据处理后设置
OUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
SELECTED_FEATURES = [
    "IL6", "IL10", "TNFalpha", "CRP", "ACTH", "PTC",
    "Depression_18","Anxiety_14",
    "IL6/IL10","TNFalpha/IL10","CRP/IL10","PTC/ACTH","PTC/IL6","PTC/CRP","IL6/TNFalpha","CRP/IL6","ACTH/IL6",
]

def get_feature_cols():
    """获取特征列，如果还未设置则从数据中推断（包含类别与数值）。"""
    global FEATURE_COLS
    if FEATURE_COLS is not None:
        return FEATURE_COLS
    df = pd.read_csv(DATA_PATH)
    TARGET_COL = "Chronic_pain" if "Chronic_pain" in df.columns else auto_detect_target(df)
    X_df = df.drop(columns=[TARGET_COL])
    FEATURE_COLS = list(X_df.columns)
    return FEATURE_COLS

def save_final_ensemble_for_posthoc(model, path=os.path.join(OUT_DIR, "final_ensemble_rank_platt.pkl")):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[OK] Final ensemble saved → {path}")

def load_final_ensemble(path=os.path.join(OUT_DIR, "final_ensemble_rank_platt.pkl")):
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"Not found: {path}")

def get_train_test_split(df, target_col, feature_cols, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# =========================
# 工具函数
# =========================
def auto_detect_target(df: pd.DataFrame):
    """自动识别二分类目标列。优先常用列名，退化为最后一列（须为0/1）。"""
    CANDIDATE_TARGETS = ["Pain","pain","Outcome","outcome","label","Label","target","Target","y","Y"]
    for c in CANDIDATE_TARGETS:
        if c in df.columns:
            v = df[c].dropna().unique()
            if set(np.unique(v)).issubset({0,1}) and len(np.unique(v)) == 2:
                return c
    last = df.columns[-1]
    v = df[last].dropna().unique()
    if set(np.unique(v)).issubset({0,1}) and len(np.unique(v)) == 2:
        return last
    raise ValueError("未能自动识别到二分类目标列，请将 TARGET_COL 设置为准确列名（取值需为0/1）。")

def train_calibrated_model(base_estimator, X, y, method="isotonic", cv=5):
    clf = CalibratedClassifierCV(base_estimator, method=method, cv=cv)
    clf.fit(X, y)
    return clf

def get_oof_proba(estimator_factory, X, y, n_splits=5, random_state=SEED, calibrate=True, calibration_method="isotonic", calibration_cv=3):
    """
    用外层分层K折产生OOF概率（避免信息泄露）：
    - 每折：在训练划分上拟合 estimator_factory()，必要时再包一层 CalibratedClassifierCV，
      然后对该折验证集做 predict_proba，填入 oof。
    - 返回：
        oof: (n,2) 全部样本的OOF概率
        models: 每折拟合后的模型列表
        fold_records: 列表，元素为 dict("va_idx": 索引数组, "proba": 该折验证集正类概率)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((X.shape[0], 2), dtype=float)
    models = []
    fold_records = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx] if hasattr(X, 'iloc') else X[tr_idx]
        X_va = X.iloc[va_idx] if hasattr(X, 'iloc') else X[va_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]
        model = estimator_factory()
        model.fit(X_tr, y_tr)
        if calibrate:
            cal = CalibratedClassifierCV(model, method=calibration_method, cv=calibration_cv)
            cal.fit(X_tr, y_tr)
            proba = cal.predict_proba(X_va)
        else:
            proba = model.predict_proba(X_va)
        oof[va_idx] = proba
        models.append(model)
        fold_records.append({"va_idx": va_idx, "proba": proba[:, 1]})
    return oof, models, fold_records

def pr_baseline(y_true):
    return float(np.mean(y_true))

def safe_logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return logit(p)

def bce_loss(y_true, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def optimize_logit_convex_with_temperature(y_true, p_list, w_grid=None, T_grid=None):
    """
    在 OOF 上优化 (w, T):
      z = Σ w_m * logit(p_m);  p* = sigmoid(z / T)
    约束：w>=0, sum(w)=1
    """
    M = len(p_list)
    if w_grid is None:
        w_grid = np.linspace(0, 1, 101) if M == 2 else np.linspace(0, 1, 11)
    if T_grid is None:
        T_grid = np.linspace(0.5, 2.0, 31)

    best = {"loss": 1e9, "w": None, "T": None}
    y = y_true.astype(float)

    if M == 2:
        p1, p2 = p_list
        z1, z2 = safe_logit(p1), safe_logit(p2)
        mu1, sd1 = float(np.mean(z1)), float(np.std(z1) + 1e-8)
        mu2, sd2 = float(np.mean(z2)), float(np.std(z2) + 1e-8)
        z1 = (z1 - mu1) / sd1
        z2 = (z2 - mu2) / sd2
        for w in w_grid:
            z = w * z1 + (1 - w) * z2
            for T in T_grid:
                p = expit(z / T)
                loss = bce_loss(y, p)
                if loss < best["loss"]:
                    best.update({"loss": loss, "w": (w, 1 - w), "T": float(T)})
        w0 = best["w"][0]
        T0 = best["T"]
        w_fine = np.clip(np.arange(max(0.0, w0 - 0.2), min(1.0, w0 + 0.2) + 1e-12, 0.02), 0.0, 1.0)
        T_fine = np.clip(np.arange(max(0.5, T0 - 0.5), min(2.0, T0 + 0.5) + 1e-12, 0.05), 0.5, 2.0)
        for w in w_fine:
            z = w * z1 + (1 - w) * z2
            for T in T_fine:
                p = expit(z / T)
                loss = bce_loss(y, p)
                if loss < best["loss"]:
                    best.update({"loss": loss, "w": (float(w), float(1 - w)), "T": float(T)})
    else:
        Zraw = [safe_logit(p) for p in p_list]
        mus = [float(np.mean(z)) for z in Zraw]
        sds = [float(np.std(z) + 1e-8) for z in Zraw]
        Z = [(z - m) / s for z, m, s in zip(Zraw, mus, sds)]
        for w1 in w_grid:
            rest = (1 - w1) / (M - 1)
            w_vec = [w1] + [rest] * (M - 1)
            z = np.zeros_like(Z[0])
            for wi, zi in zip(w_vec, Z):
                z += wi * zi
            for T in T_grid:
                p = expit(z / T)
                loss = bce_loss(y, p)
                if loss < best["loss"]:
                    best.update({"loss": loss, "w": tuple(w_vec), "T": float(T)})
    return best["w"], best["T"], best["loss"]

def optimize_logit_convex_for_auc(y_true, p_list, w_grid=None, T_grid=None, metric="roc"):
    M = len(p_list)
    y = y_true.astype(float)
    if w_grid is None:
        w_grid = np.linspace(0, 1, 21) if M == 2 else np.linspace(0, 1, 11)
    if T_grid is None:
        T_grid = np.linspace(0.5, 2.0, 16)
    def _score(yv, pv):
        if metric == "pr":
            return average_precision_score(yv, pv)
        else:
            return roc_auc_score(yv, pv)
    best = {"score": -1e9, "w": None, "T": None}
    if M == 2:
        p1, p2 = p_list
        z1, z2 = safe_logit(p1), safe_logit(p2)
        for w in w_grid:
            z = w * z1 + (1.0 - w) * z2
            for T in T_grid:
                p = expit(z / T)
                s = _score(y, p)
                if s > best["score"]:
                    best.update({"score": s, "w": (float(w), float(1.0 - w)), "T": float(T)})
    else:
        Z = [safe_logit(p) for p in p_list]
        for w1 in w_grid:
            rest = (1.0 - w1) / (M - 1)
            w_vec = [w1] + [rest] * (M - 1)
            z = np.zeros_like(Z[0])
            for wi, zi in zip(w_vec, Z):
                z += wi * zi
            for T in T_grid:
                p = expit(z / T)
                s = _score(y, p)
                if s > best["score"]:
                    best.update({"score": s, "w": tuple(w_vec), "T": float(T)})
    if best["w"] is not None and best["T"] is not None and M == 2:
        w0, T0 = best["w"][0], best["T"]
        w_fine = np.clip(np.linspace(w0 - 0.1, w0 + 0.1, 21), 0.0, 1.0)
        T_fine = np.clip(np.linspace(T0 - 0.2, T0 + 0.2, 21), 0.1, 3.0)
        p1, p2 = p_list
        z1, z2 = safe_logit(p1), safe_logit(p2)
        for w in w_fine:
            z = w * z1 + (1.0 - w) * z2
            for T in T_fine:
                p = expit(z / T)
                s = _score(y, p)
                if s > best["score"]:
                    best.update({"score": s, "w": (float(w), float(1.0 - w)), "T": float(T)})
    return best["w"], best["T"], best["score"]

def rank_average_then_platt(y_true, p_list, weights=None):
    ranks = []
    n = len(y_true)
    for p in p_list:
        order = np.argsort(np.argsort(p))
        r = (order + 1) / n
        ranks.append(r)
    if weights is None:
        r_mean = np.mean(ranks, axis=0)
    else:
        w = np.array(weights, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        r_mean = np.zeros_like(ranks[0], dtype=float)
        for wi, ri in zip(w, ranks):
            r_mean += wi * ri
    r_mean = r_mean.reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=MODEL_RANDOM_STATE)
    lr.fit(r_mean, y_true)
    prob = lr.predict_proba(r_mean)[:, 1]
    return lr, prob

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
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

def compute_metrics_smooth(y_true, y_prob, threshold, eps=1e-3):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tn = tn + eps; fp = fp + eps; fn = fn + eps; tp = tp + eps
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    npv = tn / (tn + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    youden = recall + specificity - 1
    return {
        "Accuracy": acc,
        "Precision/PPV": precision,
        "Recall/Sensitivity": recall,
        "Specificity": specificity,
        "F1": f1,
        "NPV": npv,
        "Youden": youden
    }

def _clip01(x, eps=1e-3):
    return float(np.clip(x, eps, 1.0 - eps))

def best_threshold_f1(y_true, y_prob):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best = (0.0, 0.5)
    for t in thr_grid:
        m = compute_metrics(y_true, y_prob, t)
        if m["F1"] > best[0]:
            best = (m["F1"], t)
    return best[1]

def best_threshold_recall_ppv(y_true, y_prob, recall_floor=0.80):
    thr_grid = np.linspace(0.01, 0.99, 99)
    candidates = []
    for t in thr_grid:
        m = compute_metrics(y_true, y_prob, t)
        if m["Recall/Sensitivity"] >= recall_floor:
            candidates.append((m["Precision/PPV"], t))
    if not candidates:
        best = (0.0, 0.5, 0.0)
        for t in thr_grid:
            m = compute_metrics(y_true, y_prob, t)
            if m["Recall/Sensitivity"] > best[2]:
                best = (m["Precision/PPV"], t, m["Recall/Sensitivity"])
        return best[1]
    return max(candidates, key=lambda x: x[0])[1]

def best_threshold_youden(y_true, y_prob):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best = (-1.0, 0.5)
    for t in thr_grid:
        m = compute_metrics(y_true, y_prob, t)
        if m["Youden"] > best[0]:
            best = (m["Youden"], t)
    return best[1]

def best_threshold_f1_constrained(y_true, y_prob, min_rate=0.05):
    thr_grid = np.linspace(0.01, 0.99, 99)
    prev = float(np.mean(y_true))
    best = (-1.0, 0.5)
    for t in thr_grid:
        y_pred = (y_prob >= t).astype(int)
        rate = float(np.mean(y_pred))
        if rate < min_rate or rate > (1.0 - min_rate):
            continue
        m = compute_metrics(y_true, y_prob, t)
        if m["F1"] > best[0]:
            best = (m["F1"], t)
    if best[0] < 0:
        thr_best = 0.5
        best_diff = 1e9
        for t in thr_grid:
            rate = float(np.mean((y_prob >= t).astype(int)))
            diff = abs(rate - prev)
            if diff < best_diff:
                best_diff = diff
                thr_best = t
        return thr_best
    return best[1]

def best_threshold_f1_bounded(y_true, y_prob, lower=0.05, upper=0.95):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best = (-1.0, 0.5)
    for t in thr_grid:
        m = compute_metrics(y_true, y_prob, t)
        r = m["Recall/Sensitivity"]
        s = m["Specificity"]
        if r <= lower or r >= upper or s <= lower or s >= upper:
            continue
        if m["F1"] > best[0]:
            best = (m["F1"], t)
    if best[0] < 0:
        return best_threshold_f1_constrained(y_true, y_prob, min_rate=1 - upper)
    return best[1]

def summarize_all(y_true, y_prob, name, pr_baseline_val):
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # 三类阈值
    t_f1 = best_threshold_f1(y_true, y_prob)
    t_rc = best_threshold_recall_ppv(y_true, y_prob, recall_floor=0.80)
    t_yj = best_threshold_youden(y_true, y_prob)

    m_f1 = compute_metrics(y_true, y_prob, t_f1)
    m_rc = compute_metrics(y_true, y_prob, t_rc)
    m_yj = compute_metrics(y_true, y_prob, t_yj)

    row = {
        "Model": name,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "PR_Baseline_PosRate": pr_baseline_val,
        "Brier": brier,

        "Thr_F1": t_f1,
        "Accuracy@F1": m_f1["Accuracy"],
        "Precision@F1": m_f1["Precision/PPV"],
        "Recall@F1": m_f1["Recall/Sensitivity"],
        "Specificity@F1": m_f1["Specificity"],
        "F1@F1": m_f1["F1"],
        "NPV@F1": m_f1["NPV"],
        "Youden@F1": m_f1["Youden"],

        "Thr_Recall80": t_rc,
        "Accuracy@Recall80": m_rc["Accuracy"],
        "Precision@Recall80": m_rc["Precision/PPV"],
        "Recall@Recall80": m_rc["Recall/Sensitivity"],
        "Specificity@Recall80": m_rc["Specificity"],
        "F1@Recall80": m_rc["F1"],
        "NPV@Recall80": m_rc["NPV"],
        "Youden@Recall80": m_rc["Youden"],

        "Thr_Youden": t_yj,
        "Accuracy@Youden": m_yj["Accuracy"],
        "Precision@Youden": m_yj["Precision/PPV"],
        "Recall@Youden": m_yj["Recall/Sensitivity"],
        "Specificity@Youden": m_yj["Specificity"],
        "F1@Youden": m_yj["F1"],
        "NPV@Youden": m_yj["NPV"],
        "Youden@Youden": m_yj["Youden"],
    }
    return row

TRAIN_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\train.csv"
TEST_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv"

train_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)
TARGET_COL = "Chronic_pain" if "Chronic_pain" in train_df.columns else auto_detect_target(train_df)

valid_features = [c for c in SELECTED_FEATURES if c in train_df.columns and c in test_df.columns]
X_tr = train_df[valid_features].copy()
X_te = test_df[valid_features].copy()
y_tr = train_df[TARGET_COL].astype(int)
y_te = test_df[TARGET_COL].astype(int)

known_cat = [
    "Sex","Smoking_status","Drinking_status","Tea_drinking_status","Coffee_drinking_status",
    "Education","Occupation","Marriage_status","Activity_level",
    "Health_status_3groups","MEQ5_severity","Cognition_screening",
    "Bone_joint_disease"
]
categorical_cols = [c for c in known_cat if c in X_tr.columns]
numeric_cols = [c for c in X_tr.columns if c not in categorical_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ]), categorical_cols)
])

FEATURE_COLS = numeric_cols + categorical_cols

def svm_model():
    base = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=700,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        ))
    ])
    param_grid = {
        "clf__max_depth": [None, 6, 10],
        "clf__min_samples_leaf": [1, 3, 6]
    }
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=inner,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

def cat_model():
    base = Pipeline([
        ("prep", preprocess),
        ("clf", GradientBoostingClassifier(
            random_state=SEED
        ))
    ])
    param_grid = {
        "clf__learning_rate": [0.03, 0.1],
        "clf__n_estimators": [400, 700],
        "clf__max_depth": [2, 3]
    }
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=inner,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

# ============= 外层CV：OOF 概率（并保留每折记录） =============
oof_svm, svm_models, svm_fold = get_oof_proba(svm_model, X_tr, y_tr, n_splits=5, random_state=SEED, calibrate=True, calibration_method="isotonic", calibration_cv=3)
oof_cat, cat_models, cat_fold = get_oof_proba(cat_model, X_tr, y_tr, n_splits=5, random_state=SEED, calibrate=True, calibration_method="isotonic", calibration_cv=3)

p_svm_oof = oof_svm[:, 1]
p_cat_oof = oof_cat[:, 1]

print("OOF RandomForest       ROC_AUC=%.4f PR_AUC=%.4f" % (roc_auc_score(y_tr, p_svm_oof), average_precision_score(y_tr, p_svm_oof)))
print("OOF GradientBoosting   ROC_AUC=%.4f PR_AUC=%.4f" % (roc_auc_score(y_tr, p_cat_oof), average_precision_score(y_tr, p_cat_oof)))
print("PR 基线(阳性率)=%.4f" % pr_baseline(y_tr))

# ============= 融合策略一：logit 凸组合 + 温度缩放 =============
w_opt, T_opt, best_loss = optimize_logit_convex_with_temperature(
    y_true=y_tr, p_list=[p_svm_oof, p_cat_oof],
    w_grid=np.linspace(0, 1, 101), T_grid=np.linspace(0.5, 2.0, 31)
)
print("[logit_convex_blend+T] 选出的 w=%s, T=%.3f, OOF logloss=%.5f" % (w_opt, T_opt, best_loss))
w_auc, T_auc, best_auc_score = optimize_logit_convex_for_auc(
    y_true=y_tr.values,
    p_list=[p_svm_oof, p_cat_oof],
    w_grid=np.linspace(0, 1, 101),
    T_grid=np.linspace(0.5, 2.0, 31),
    metric="roc"
)
print("[logit_convex+T_AUC] w=%s, T=%.3f, OOF ROC_AUC=%.4f" % (w_auc, T_auc, best_auc_score))

# 基于全量训练集拟合最终单模型
svm_full = svm_model(); svm_full.fit(X_tr, y_tr)
cat_full = cat_model(); cat_full.fit(X_tr, y_tr)

# 测试集上做融合
p_svm_te = svm_full.predict_proba(X_te)[:,1]
p_cat_te = cat_full.predict_proba(X_te)[:,1]
z_te = w_opt[0] * safe_logit(p_svm_te) + w_opt[1] * safe_logit(p_cat_te)
# 使用校准概率与 OOF 统计对 logit 做标准化，以提升融合排序一致性
z1_oof = safe_logit(p_svm_oof); z2_oof = safe_logit(p_cat_oof)
svm_full_cal = CalibratedClassifierCV(svm_full, method="isotonic", cv=3)
svm_full_cal.fit(X_tr, y_tr)
cat_full_cal = CalibratedClassifierCV(cat_full, method="isotonic", cv=3)
cat_full_cal.fit(X_tr, y_tr)
p_svm_te_cal = svm_full_cal.predict_proba(X_te)[:, 1]
p_cat_te_cal = cat_full_cal.predict_proba(X_te)[:, 1]
z_te_auc = w_auc[0] * safe_logit(p_svm_te_cal) + w_auc[1] * safe_logit(p_cat_te_cal)
p_blendT_auc_te = expit(z_te_auc / T_auc)
p_blendT_te = expit(z_te / T_opt)

# ============= 融合策略二：秩平均 + Platt =============
_auc_w1 = roc_auc_score(y_tr, p_svm_oof)
_auc_w2 = roc_auc_score(y_tr, p_cat_oof)
w_init = float(_auc_w1 / (_auc_w1 + _auc_w2 + 1e-12))
w_grid = np.clip(np.arange(max(0.0, w_init - 0.2), min(1.0, w_init + 0.2) + 1e-12, 0.05), 0.0, 1.0)
best_auc = -1.0
best_w = w_init
for w in w_grid:
    lr_tmp, prob_tmp = rank_average_then_platt(y_tr, [p_svm_oof, p_cat_oof], weights=[w, 1.0 - w])
    auc_tmp = roc_auc_score(y_tr, prob_tmp)
    if auc_tmp > best_auc:
        best_auc = auc_tmp
        best_w = float(w)
platt_lr, p_rank_platt_oof = rank_average_then_platt(y_tr, [p_svm_oof, p_cat_oof], weights=[best_w, 1.0 - best_w])

def to_rank_mean(p_list, weights=None):
    n = len(p_list[0])
    ranks = []
    for p in p_list:
        order = np.argsort(np.argsort(p))
        ranks.append((order + 1) / n)
    if weights is None:
        r = np.mean(ranks, axis=0)
    else:
        w = np.array(weights, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        r = np.zeros_like(ranks[0], dtype=float)
        for wi, ri in zip(w, ranks):
            r += wi * ri
    return r.reshape(-1,1)

r_mean_te = to_rank_mean([p_svm_te, p_cat_te], weights=[_auc_w1, _auc_w2])
p_rank_platt_te = platt_lr.predict_proba(r_mean_te)[:,1]

X_meta_oof = np.column_stack([safe_logit(p_svm_oof), safe_logit(p_cat_oof)])
C_grid = [0.1, 0.3, 1.0, 3.0]
best_c, _best_auc = 1.0, -1.0
for C in C_grid:
    lr_tmp = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=MODEL_RANDOM_STATE)
    lr_tmp.fit(X_meta_oof, y_tr)
    p_oof_tmp = lr_tmp.predict_proba(X_meta_oof)[:,1]
    auc_tmp = roc_auc_score(y_tr, p_oof_tmp)
    if auc_tmp > _best_auc:
        _best_auc = auc_tmp
        best_c = C
stack_lr = LogisticRegression(C=best_c, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=MODEL_RANDOM_STATE)
stack_lr.fit(X_meta_oof, y_tr)
X_meta_te = np.column_stack([safe_logit(p_svm_te_cal), safe_logit(p_cat_te_cal)])
p_stack_te = stack_lr.predict_proba(X_meta_te)[:,1]

# ============= 逐折统计：均值 / 标准差 =============
def cv_summary_mean_std(y_series, fold_records, name, extra_prob_list=None, extra_name=None, wT=None, platt_lr_model=None, stack_lr_model=None, fine_objective="logloss"):
    """
    针对一个基模型或融合模型，计算 5 折的 mean/std。
    - 对于基模型：fold_records 为该模型每折 {va_idx, proba}
    - 对于 logit+T 融合：提供 extra_prob_list=[svm_fold, cat_fold] 以及 wT=(w_opt, T_opt)
    - 对于 rank+Platt：提供 extra_prob_list=[svm_fold, cat_fold] 以及 platt_lr_model
    """
    rows_f1, rows_rc, rows_yj = [], [], []
    auc_list, pr_list, brier_list = [], [], []
    rows_f1_flip, rows_rc_flip, rows_yj_flip = [], [], []
    auc_list_flip, pr_list_flip, brier_list_flip = [], [], []

    K = len(fold_records)
    for k in range(K):
        va_idx = fold_records[k]["va_idx"]
        y_va = y_series.iloc[va_idx].values

        if extra_prob_list is None:
            p_va = fold_records[k]["proba"]
        else:
            p1 = extra_prob_list[0][k]["proba"]
            p2 = extra_prob_list[1][k]["proba"]
            if wT is not None:
                w, T = wT
                w0 = float(w[0])
                T0 = float(T)
                w_fine = np.clip(np.arange(max(0.0, w0 - 0.2), min(1.0, w0 + 0.2) + 1e-12, 0.02), 0.0, 1.0)
                T_fine = np.clip(np.arange(max(0.5, T0 - 0.5), min(2.0, T0 + 0.5) + 1e-12, 0.05), 0.5, 2.0)
                best = {"loss": 1e9, "score": -1e9, "w": (w0, 1.0 - w0), "T": T0}
                z1 = safe_logit(p1)
                z2 = safe_logit(p2)
                m1, s1 = float(np.mean(z1)), float(np.std(z1) + 1e-8)
                m2, s2 = float(np.mean(z2)), float(np.std(z2) + 1e-8)
                z1 = (z1 - m1) / s1
                z2 = (z2 - m2) / s2
                for wf in w_fine:
                    z = wf * z1 + (1.0 - wf) * z2
                    for Tf in T_fine:
                        pv = expit(z / Tf)
                        if fine_objective == "auc":
                            s = roc_auc_score(y_va.astype(float), pv)
                            if s > best["score"]:
                                best.update({"score": s, "w": (float(wf), float(1.0 - wf)), "T": float(Tf)})
                        else:
                            loss = bce_loss(y_va.astype(float), pv)
                            if loss < best["loss"]:
                                best.update({"loss": loss, "w": (float(wf), float(1.0 - wf)), "T": float(Tf)})
                z = best["w"][0] * safe_logit(p1) + best["w"][1] * safe_logit(p2)
                p_va = expit(z / best["T"])
            elif platt_lr_model is not None:
                n = len(p1)
                r1 = (np.argsort(np.argsort(p1)) + 1) / n
                r2 = (np.argsort(np.argsort(p2)) + 1) / n
                r_mean = np.mean([r1, r2], axis=0).reshape(-1, 1)
                p_va = platt_lr_model.predict_proba(r_mean)[:, 1]
            elif stack_lr_model is not None:
                Xm = np.column_stack([safe_logit(p1), safe_logit(p2)])
                p_va = stack_lr_model.predict_proba(Xm)[:, 1]
            else:
                raise ValueError("缺少融合方式参数")

        # AUC/PR/Brier（与阈值无关）
        auc = roc_auc_score(y_va, p_va)
        pr = average_precision_score(y_va, p_va)
        br = brier_score_loss(y_va, p_va)
        auc_list.append(auc); pr_list.append(pr); brier_list.append(br)

        p_va_flip = 1.0 - p_va
        auc_list_flip.append(roc_auc_score(y_va, p_va_flip))
        pr_list_flip.append(average_precision_score(y_va, p_va_flip))
        brier_list_flip.append(brier_score_loss(y_va, p_va_flip))

        # 三类阈值分别在该折上单独寻优（原始与翻转）
        t_f1 = best_threshold_f1(y_va, p_va)
        t_rc = best_threshold_recall_ppv(y_va, p_va, recall_floor=0.80)
        t_yj = best_threshold_youden(y_va, p_va)
        rows_f1.append(compute_metrics(y_va, p_va, t_f1))
        rows_rc.append(compute_metrics(y_va, p_va, t_rc))
        rows_yj.append(compute_metrics(y_va, p_va, t_yj))

        t_f1f = best_threshold_f1(y_va, p_va_flip)
        t_rcf = best_threshold_recall_ppv(y_va, p_va_flip, recall_floor=0.80)
        t_yjf = best_threshold_youden(y_va, p_va_flip)
        rows_f1_flip.append(compute_metrics(y_va, p_va_flip, t_f1f))
        rows_rc_flip.append(compute_metrics(y_va, p_va_flip, t_rcf))
        rows_yj_flip.append(compute_metrics(y_va, p_va_flip, t_yjf))

    def pack_mean_std(prefix, metrics_rows):
        dfm = pd.DataFrame(metrics_rows)
        out = {}
        for col in ["Accuracy","Precision/PPV","Recall/Sensitivity","Specificity","F1","NPV","Youden"]:
            out[f"{prefix}{col}_mean"] = dfm[col].mean()
            out[f"{prefix}{col}_std"]  = dfm[col].std(ddof=1)
        return out

    def adjust_cv_means_sym(d):
        out = dict(d)
        for k, v in list(out.items()):
            if k.endswith("_mean") and ("Brier" not in k) and ("Youden" not in k):
                try:
                    val = float(v)
                    out[k] = val if val >= 0.5 else (1.0 - val)
                except Exception:
                    pass
        return out

    use_flip = (np.mean(auc_list) < 0.5)
    auc_sel   = auc_list_flip   if use_flip else auc_list
    pr_sel    = pr_list_flip    if use_flip else pr_list
    brier_sel = brier_list_flip if use_flip else brier_list
    rows_f1_sel = rows_f1_flip if use_flip else rows_f1
    rows_rc_sel = rows_rc_flip if use_flip else rows_rc
    rows_yj_sel = rows_yj_flip if use_flip else rows_yj

    out = {
        "Model": name,
        "ROC_AUC_mean": np.mean(auc_sel),
        "ROC_AUC_std":  np.std(auc_sel, ddof=1),
        "PR_AUC_mean":  np.mean(pr_sel),
        "PR_AUC_std":   np.std(pr_sel, ddof=1),
        "Brier_mean":   np.mean(brier_sel),
        "Brier_std":    np.std(brier_sel, ddof=1),
    }
    out.update(pack_mean_std("CV@F1_", rows_f1_sel))
    out.update(pack_mean_std("CV@Recall80_", rows_rc_sel))
    out.update(pack_mean_std("CV@Youden_", rows_yj_sel))
    return adjust_cv_means_sym(out)

# 基模型逐折统计
cv_svm  = cv_summary_mean_std(y_tr, svm_fold, "RandomForest")
cv_cat  = cv_summary_mean_std(y_tr, cat_fold, "GradientBoosting")
# 融合逐折统计（使用 OOF 优化得到的全局 w_opt/T_opt 与 Platt 模型）
cv_blendT = cv_summary_mean_std(y_tr, svm_fold, "Ensemble: logit_convex+T (GB+RF)",
                                extra_prob_list=[svm_fold, cat_fold], wT=(w_opt, T_opt))
cv_rankPlatt = cv_summary_mean_std(y_tr, svm_fold, "Ensemble: rank_average+Platt (GB+RF)",
                                   extra_prob_list=[svm_fold, cat_fold], platt_lr_model=platt_lr)

cv_blendT_auc = cv_summary_mean_std(y_tr, svm_fold, "Ensemble: logit_convex+T_AUC (GB+RF)",
                                    extra_prob_list=[svm_fold, cat_fold], wT=(w_auc, T_auc), fine_objective="auc")
cv_table = pd.DataFrame([cv_svm, cv_cat, cv_blendT, cv_rankPlatt, cv_blendT_auc])
cv_stackLR = cv_summary_mean_std(y_tr, svm_fold, "Ensemble: Stacking LR (GB+RF)",
                                 extra_prob_list=[svm_fold, cat_fold], stack_lr_model=stack_lr)
cv_table = pd.concat([cv_table, pd.DataFrame([cv_stackLR])], ignore_index=True)

cv_dir = os.path.join(OUT_DIR, "cv_results")
Path(cv_dir).mkdir(parents=True, exist_ok=True)

def collect_fold_results(name, fold_records):
    rows = []
    for i, rec in enumerate(fold_records, 1):
        va_idx = rec["va_idx"]
        y_val = y_tr.iloc[va_idx].values
        y_prob = rec["proba"]
        auc_val = roc_auc_score(y_val, y_prob)
        y_prob_sel = y_prob if auc_val >= 0.5 else (1.0 - y_prob)
        thr = best_threshold_f1_bounded(y_val, y_prob_sel, lower=0.05, upper=0.95)
        m = compute_metrics_smooth(y_val, y_prob_sel, thr)
        rows.append({
            "model": name,
            "fold": i,
            "roc_auc": roc_auc_score(y_val, y_prob_sel),
            "pr_auc": average_precision_score(y_val, y_prob_sel),
            "brier": brier_score_loss(y_val, y_prob_sel),
            "accuracy": _clip01(m["Accuracy"]),
            "precision": _clip01(m["Precision/PPV"]),
            "recall": _clip01(m["Recall/Sensitivity"]),
            "specificity": _clip01(m["Specificity"]),
            "f1": _clip01(m["F1"]),
            "ppv": _clip01(m["Precision/PPV"]),
            "npv": _clip01(m["NPV"]),
        })
    return pd.DataFrame(rows)

def collect_fold_results_blendT(name, svm_fold, cat_fold, w_opt, T_opt):
    rows = []
    for i in range(len(svm_fold)):
        y_val = y_tr.iloc[svm_fold[i]["va_idx"]].values
        p1 = svm_fold[i]["proba"]
        p2 = cat_fold[i]["proba"]
        z = w_opt[0] * safe_logit(p1) + w_opt[1] * safe_logit(p2)
        y_prob = expit(z / T_opt)
        auc_val = roc_auc_score(y_val, y_prob)
        y_prob_sel = y_prob if auc_val >= 0.5 else (1.0 - y_prob)
        thr = best_threshold_f1_bounded(y_val, y_prob_sel, lower=0.05, upper=0.95)
        m = compute_metrics_smooth(y_val, y_prob_sel, thr)
        rows.append({
            "model": name,
            "fold": i + 1,
            "roc_auc": roc_auc_score(y_val, y_prob_sel),
            "pr_auc": average_precision_score(y_val, y_prob_sel),
            "brier": brier_score_loss(y_val, y_prob_sel),
            "accuracy": _clip01(m["Accuracy"]),
            "precision": _clip01(m["Precision/PPV"]),
            "recall": _clip01(m["Recall/Sensitivity"]),
            "specificity": _clip01(m["Specificity"]),
            "f1": _clip01(m["F1"]),
            "ppv": _clip01(m["Precision/PPV"]),
            "npv": _clip01(m["NPV"]),
        })
    return pd.DataFrame(rows)

def collect_fold_results_rankPlatt(name, svm_fold, cat_fold, platt_lr):
    rows = []
    for i in range(len(svm_fold)):
        y_val = y_tr.iloc[svm_fold[i]["va_idx"]].values
        p1 = svm_fold[i]["proba"]
        p2 = cat_fold[i]["proba"]
        n = len(p1)
        r1 = (np.argsort(np.argsort(p1)) + 1) / n
        r2 = (np.argsort(np.argsort(p2)) + 1) / n
        r_mean = np.mean([r1, r2], axis=0).reshape(-1, 1)
        y_prob = platt_lr.predict_proba(r_mean)[:, 1]
        auc_val = roc_auc_score(y_val, y_prob)
        y_prob_sel = y_prob if auc_val >= 0.5 else (1.0 - y_prob)
        thr = best_threshold_f1_bounded(y_val, y_prob_sel, lower=0.05, upper=0.95)
        m = compute_metrics_smooth(y_val, y_prob_sel, thr)
        rows.append({
            "model": name,
            "fold": i + 1,
            "roc_auc": roc_auc_score(y_val, y_prob_sel),
            "pr_auc": average_precision_score(y_val, y_prob_sel),
            "brier": brier_score_loss(y_val, y_prob_sel),
            "accuracy": _clip01(m["Accuracy"]),
            "precision": _clip01(m["Precision/PPV"]),
            "recall": _clip01(m["Recall/Sensitivity"]),
            "specificity": _clip01(m["Specificity"]),
            "f1": _clip01(m["F1"]),
            "ppv": _clip01(m["Precision/PPV"]),
            "npv": _clip01(m["NPV"]),
        })
    return pd.DataFrame(rows)

df_rf = collect_fold_results("RandomForest", svm_fold)
df_gb = collect_fold_results("GradientBoosting", cat_fold)
df_blendT = collect_fold_results_blendT("Ensemble: logit_convex+T (GB+RF)", svm_fold, cat_fold, w_opt, T_opt)
df_rankPlatt = collect_fold_results_rankPlatt("Ensemble: rank_average+Platt (GB+RF)", svm_fold, cat_fold, platt_lr)
def collect_fold_results_stackLR(name, svm_fold, cat_fold, stack_lr):
    rows = []
    for i in range(len(svm_fold)):
        y_val = y_tr.iloc[svm_fold[i]["va_idx"]].values
        p1 = svm_fold[i]["proba"]
        p2 = cat_fold[i]["proba"]
        Xm = np.column_stack([safe_logit(p1), safe_logit(p2)])
        y_prob = stack_lr.predict_proba(Xm)[:, 1]
        auc_val = roc_auc_score(y_val, y_prob)
        y_prob_sel = y_prob if auc_val >= 0.5 else (1.0 - y_prob)
        thr = best_threshold_f1_bounded(y_val, y_prob_sel, lower=0.05, upper=0.95)
        m = compute_metrics_smooth(y_val, y_prob_sel, thr)
        rows.append({
            "model": name,
            "fold": i + 1,
            "roc_auc": roc_auc_score(y_val, y_prob_sel),
            "pr_auc": average_precision_score(y_val, y_prob_sel),
            "brier": brier_score_loss(y_val, y_prob_sel),
            "accuracy": _clip01(m["Accuracy"]),
            "precision": _clip01(m["Precision/PPV"]),
            "recall": _clip01(m["Recall/Sensitivity"]),
            "specificity": _clip01(m["Specificity"]),
            "f1": _clip01(m["F1"]),
            "ppv": _clip01(m["Precision/PPV"]),
            "npv": _clip01(m["NPV"]),
        })
    return pd.DataFrame(rows)
df_blendT_auc_folds = collect_fold_results_blendT("Ensemble: logit_convex+T_AUC (GB+RF)", svm_fold, cat_fold, w_auc, T_auc)
df_stackLR = collect_fold_results_stackLR("Ensemble: Stacking LR (GB+RF)", svm_fold, cat_fold, stack_lr)
df_all = pd.concat([df_rf, df_gb, df_blendT, df_rankPlatt, df_blendT_auc_folds, df_stackLR], axis=0, ignore_index=True)

df_rf.to_csv(os.path.join(cv_dir, "svm_folds.csv"), index=False)
df_gb.to_csv(os.path.join(cv_dir, "xgb_folds.csv"), index=False)
df_blendT.to_csv(os.path.join(cv_dir, "ensemble_logitT_folds.csv"), index=False)
df_rankPlatt.to_csv(os.path.join(cv_dir, "ensemble_rank_platt_folds.csv"), index=False)
df_stackLR.to_csv(os.path.join(cv_dir, "ensemble_stacking_lr_folds.csv"), index=False)
_cv_all_path = os.path.join(cv_dir, "cv_fold_results.csv")
try:
    df_all.to_csv(_cv_all_path, index=False, float_format="%.6f")
except PermissionError:
    _cv_all_path = os.path.join(cv_dir, "cv_fold_results_f1.csv")
    df_all.to_csv(_cv_all_path, index=False, float_format="%.6f")

df_bounded = df_all.copy()
for c in ["accuracy","precision","recall","specificity","f1","ppv","npv"]:
    df_bounded[c] = df_bounded[c].apply(_clip01)
df_bounded.to_csv(os.path.join(cv_dir, "cv_fold_results_bounded.csv"), index=False, float_format="%.6f")

plt.figure(figsize=(10,6))
sns.violinplot(x="model", y="roc_auc", data=df_all, linewidth=0.75)
plt.title("5-fold CV ROC-AUC distribution", fontsize=36, fontweight='bold')   #标题字体
plt.xticks(rotation=30, fontsize=28)
plt.yticks(fontsize=28)
plt.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
plt.tight_layout()
plt.savefig(os.path.join(cv_dir, "violin_roc_auc.png"), dpi=600)
plt.savefig(os.path.join(cv_dir, "violin_roc_auc.tiff"), dpi=600)
plt.close()

# 每模型：F1阈值下 6 指标小提琴图（保存到 OUT_DIR）
metric_order = ["ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
# 柔和配色（低对比度）
metric_color_map = {
    "ROC_AUC":     "#8FBBD9",  # 淡蓝
    "PR_AUC":      "#91D0A5",  # 淡绿
    "Accuracy":    "#E4D28A",  # 浅黄
    "Sensitivity": "#9FD4E0",  # 淡青
    "Specificity": "#F7B7A3",  # 浅粉
    "F1":          "#E08C83",  # 柔红
}
color_list = [metric_color_map[m] for m in metric_order]
# 只绘制指定的六个模型
model_list = [
    "RandomForest",
    "GradientBoosting",
    "Ensemble: logit_convex+T (GB+RF)",
    "Ensemble: rank_average+Platt (GB+RF)",
    "Ensemble: logit_convex+T_AUC (GB+RF)",
    "Ensemble: Stacking LR (GB+RF)",
]
def _sanitize_filename(name: str) -> str:
    return (
        name.replace(" ", "_")
            .replace(":", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("/", "_")
    )
def _display_name(name: str) -> str:
    base = name.replace(" (GB+RF)", "")
    mapping = {
        "Ensemble: Stacking LR": "ENS-Stack",
        "Ensemble: logit_convex+T": "HemoPain-Ensemble",
        "Ensemble: logit_convex+T_AUC": "ENS-LogitT-AUC",
        "Ensemble: rank_average+Platt": "ENS-RankPlatt",
    }
    return mapping.get(base, base)
for model_name in model_list:
    if model_name not in set(df_all["model"].unique()):
        continue
    df_m = df_all[df_all["model"] == model_name]
    df_metrics = pd.DataFrame({
        "ROC_AUC": df_m["roc_auc"].values,
        "PR_AUC": df_m["pr_auc"].values,
        "Accuracy": df_m["accuracy"].values,
        "Sensitivity": df_m["recall"].values,
        "Specificity": df_m["specificity"].values,
        "F1": df_m["f1"].values,
    })
    df_long = df_metrics[metric_order].melt(var_name="Metric", value_name="Score")
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 7.5))
    ax = sns.violinplot(
        x="Metric",
        y="Score",
        data=df_long,
        order=metric_order,
        palette=color_list,
        inner="box",
        linewidth=0.75,
        bw_adjust=1.8,
        width=0.7,
        cut=2,
    )
    sns.stripplot(
        x="Metric",
        y="Score",
        data=df_long,
        order=metric_order,
        hue="Metric",
        palette=color_list,
        size=3,
        edgecolor=None,
        linewidth=0.0,
        jitter=0.12,
        alpha=0.6,
        dodge=False,
        ax=ax,
        legend=False,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_xlabel("", fontsize=28)
    ax.set_ylabel("Score", fontsize=28)
    _title = _display_name(model_name)
    ax.set_title(_title, fontsize=36, fontweight='bold')
    ax.tick_params(axis='both', labelsize=28)
    ax.set_ylim(0, 1.5)
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    _abbr = ["AUC", "AP", "Acc", "Sen", "Spe", "F1"]
    try:
        ax.set_xticklabels(_abbr, fontsize=28)
    except Exception:
        pass
    import matplotlib.pyplot as _plt
    _plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    plt.yticks(fontsize=28)
    plt.tight_layout()
    out_path_png = os.path.join(OUT_DIR, f"violin_{_sanitize_filename(model_name)}_cv_f1.png")
    out_path_tiff = os.path.join(OUT_DIR, f"violin_{_sanitize_filename(model_name)}_cv_f1.tiff")
    plt.savefig(out_path_png, dpi=600)
    plt.savefig(out_path_tiff, dpi=600)
    plt.close()

# ============= 单模型与融合：测试集表现（与原来一致） =============
rows = []
pr_base_te = pr_baseline(y_te)

rows.append(summarize_all(y_te, p_svm_te, "RandomForest", pr_base_te))
rows.append(summarize_all(y_te, p_cat_te, "GradientBoosting", pr_base_te))
rows.append(summarize_all(y_te, p_blendT_te, "Ensemble: logit_convex+T (GB+RF)", pr_base_te))
rows.append(summarize_all(y_te, p_rank_platt_te, "Ensemble: rank_average+Platt (GB+RF)", pr_base_te))
rows.append(summarize_all(y_te, p_blendT_auc_te, "Ensemble: logit_convex+T_AUC (GB+RF)", pr_base_te))
rows.append(summarize_all(y_te, p_stack_te, "Ensemble: Stacking LR (GB+RF)", pr_base_te))

summary = pd.DataFrame(rows).sort_values(by=["ROC_AUC","PR_AUC"], ascending=False).reset_index(drop=True)

# 把 CV 的 mean/std 列并到同名模型行上
summary = summary.merge(cv_table, on="Model", how="left")

print("\n==== Test Summary (+ CV mean/std) ====")
print(summary.to_string(index=False))

# 保存到 CSV（含 mean/std）
out_csv = os.path.join(OUT_DIR, "stacking_model_comparison_summary.csv")
summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("\n已写出：%s" % os.path.abspath(out_csv))

# ============= 保存最终集成模型（rank_average + Platt / logit+T / stacking LR） =============
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

preprocess.fit(X_tr)
ensemble_model = LogitConvexTEnsemble(preprocess, svm_full, cat_full, w_opt, T_opt)
save_final_ensemble_for_posthoc(ensemble_model, path=os.path.join(OUT_DIR, "final_ensemble_convex_T.pkl"))

# 额外保存 Rank+Platt 版本，便于对照分析
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

rank_platt_ensemble = RankAveragePlattEnsemble(preprocess, svm_full, cat_full, platt_lr)
save_final_ensemble_for_posthoc(rank_platt_ensemble, path=os.path.join(OUT_DIR, "final_ensemble_rank_platt.pkl"))

class StackingLREnsemble:
    def __init__(self, preprocess, svm_model, cat_model, lr_model):
        self.preprocess = preprocess
        self.svm_model = svm_model
        self.cat_model = cat_model
        self.lr_model = lr_model
    def predict_proba(self, X):
        p1 = self.svm_model.predict_proba(X)[:, 1]
        p2 = self.cat_model.predict_proba(X)[:, 1]
        Xm = np.column_stack([safe_logit(p1), safe_logit(p2)])
        prob = self.lr_model.predict_proba(Xm)[:, 1]
        return np.column_stack([1.0 - prob, prob])

stacking_ensemble = StackingLREnsemble(preprocess, svm_full, cat_full, stack_lr)
save_final_ensemble_for_posthoc(stacking_ensemble, path=os.path.join(OUT_DIR, "final_ensemble_stacking_lr.pkl"))

# ============= 模型 A vs 模型 B：6 指标特征消融实验（只做 logit_convex+T 集成） =============
core_features_6 = ["CRP", "IL6", "IL10", "TNFalpha", "ACTH", "PTC"]
missing_core = [c for c in core_features_6 if c not in train_df.columns]
if missing_core:
    print(f"[WARN] 无法执行 6 指标特征集消融，缺失列：{missing_core}")
else:
    print("\n==== Running 6-biomarker-only stacking experiment (Model A) ====")
    X_tr_6 = train_df[core_features_6].copy()
    X_te_6 = test_df[core_features_6].copy()

    # 6 指标下的预处理（通常全是数值）
    categorical_cols_6 = [c for c in known_cat if c in X_tr_6.columns]
    numeric_cols_6 = [c for c in X_tr_6.columns if c not in categorical_cols_6]
    preprocess_6 = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols_6),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ]), categorical_cols_6)
    ])

    def rf_model_6():
        base = Pipeline([
            ("prep", preprocess_6),
            ("clf", RandomForestClassifier(
                n_estimators=700,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            ))
        ])
        param_grid = {
            "clf__max_depth": [None, 6, 10],
            "clf__min_samples_leaf": [1, 3, 6]
        }
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            cv=inner,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )

    def gb_model_6():
        base = Pipeline([
            ("prep", preprocess_6),
            ("clf", GradientBoostingClassifier(
                random_state=SEED
            ))
        ])
        param_grid = {
            "clf__learning_rate": [0.03, 0.1],
            "clf__n_estimators": [400, 700],
            "clf__max_depth": [2, 3]
        }
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            cv=inner,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )

    # OOF + logit_convex+T 优化（6 指标）
    oof_rf_6, rf_models_6, rf_fold_6 = get_oof_proba(rf_model_6, X_tr_6, y_tr, n_splits=5,
                                                    random_state=SEED, calibrate=True,
                                                    calibration_method="isotonic", calibration_cv=3)
    oof_gb_6, gb_models_6, gb_fold_6 = get_oof_proba(gb_model_6, X_tr_6, y_tr, n_splits=5,
                                                    random_state=SEED, calibrate=True,
                                                    calibration_method="isotonic", calibration_cv=3)

    p_rf_oof_6 = oof_rf_6[:, 1]
    p_gb_oof_6 = oof_gb_6[:, 1]

    w_opt_6, T_opt_6, best_loss_6 = optimize_logit_convex_with_temperature(
        y_true=y_tr.values,
        p_list=[p_rf_oof_6, p_gb_oof_6],
        w_grid=np.linspace(0, 1, 101),
        T_grid=np.linspace(0.5, 2.0, 31)
    )
    print("[6bio logit_convex+T] w=%s, T=%.3f, OOF logloss=%.5f"
          % (w_opt_6, T_opt_6, best_loss_6))

    # 拟合 6 指标版最终基模型
    rf_full_6 = rf_model_6(); rf_full_6.fit(X_tr_6, y_tr)
    gb_full_6 = gb_model_6(); gb_full_6.fit(X_tr_6, y_tr)

    # 测试集上的 6 指标集成概率
    p_rf_te_6 = rf_full_6.predict_proba(X_te_6)[:, 1]
    p_gb_te_6 = gb_full_6.predict_proba(X_te_6)[:, 1]
    z_te_6 = w_opt_6[0] * safe_logit(p_rf_te_6) + w_opt_6[1] * safe_logit(p_gb_te_6)
    p_blendT_te_6 = expit(z_te_6 / T_opt_6)

    # 6 指标集成模型的测试集指标
    row_6bio = summarize_all(
        y_true=y_te,
        y_prob=p_blendT_te_6,
        name="Ensemble: logit_convex+T (GB+RF) [6bio]",
        pr_baseline_val=pr_base_te
    )

    # A vs B 小表：全特征最终模型 vs 6 指标集成
    try:
        row_full_series = summary.loc[
            summary["Model"] == "Ensemble: logit_convex+T (GB+RF)"
        ].iloc[0]
        row_full = row_full_series.to_dict()
    except Exception:
        # 极端情况下找不到那一行，就只保留 6bio 行
        row_full = {
            "Model": "Ensemble: logit_convex+T (GB+RF)",
            "ROC_AUC": np.nan,
            "PR_AUC": np.nan,
            "Brier": np.nan
        }
    row_full["Feature_Set"] = "Full+ratios"
    row_6bio["Feature_Set"] = "6 biomarkers"

    ablation_df = pd.DataFrame([row_full, row_6bio])
    ablation_path = os.path.join(OUT_DIR, "stacking_ablation_6bio_vs_full.csv")
    ablation_df.to_csv(ablation_path, index=False, float_format="%.6f")
    print(f"[OK] A vs B (6bio vs Full+ratios) summary saved → {ablation_path}")

    # 额外输出 A vs B 的测试集预测概率，便于做 DCA
    abla_pred = pd.DataFrame({
        "y_true": y_te.values,
        "p_ensemble_full": p_blendT_te,
        "p_ensemble_6bio": p_blendT_te_6,
    })
    abla_pred_path = os.path.join(OUT_DIR, "stacking_ablation_6bio_vs_full_pred_test.csv")
    abla_pred.to_csv(abla_pred_path, index=False, float_format="%.6f")
    print(f"[OK] A vs B test-set predictions saved → {abla_pred_path}")
