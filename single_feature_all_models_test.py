# -*- coding: utf-8 -*-
"""
单特征批量建模脚本（测试集结果，F1-opt only）

用途：
- 对 15 个血液特征逐一做单特征预测 Chronic_pain
- 覆盖基础模型 + 集成模型
- 输出类似 S2 的测试集结果表（仅保留 F1 最大化阈值）
- 额外保存每个 模型-特征 的测试集预测概率

放置位置：与 competition.py / stacking_competition.py 同目录
运行方式：python single_feature_all_models_test.py
"""

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# =========================
# 路径与配置
# =========================
TRAIN_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\train.csv"
TEST_CSV  = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv"
OUT_DIR   = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\single_feature_models"

TARGET_COL = "Chronic_pain"
SEED = 42
CV_N_SPLITS = 5

# 15 个血液特征
SINGLE_FEATURES = [
    "CRP", "IL6", "IL10", "TNFalpha", "ACTH", "PTC",
    "IL6/IL10", "TNFalpha/IL10", "CRP/IL10", "PTC/ACTH",
    "PTC/IL6", "PTC/CRP", "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6"
]

# 若未安装对应库，则自动跳过
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


# =========================
# 工具函数
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_logit(p, eps=1e-5):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return logit(p)


def bce_loss(y_true, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y_true, dtype=float)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    youden = recall + specificity - 1.0

    return {
        "Accuracy": acc,
        "Precision/PPV": precision,
        "Recall/Sensitivity": recall,
        "Specificity": specificity,
        "F1": f1,
        "NPV": npv,
        "Youden": youden,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def best_threshold_f1(y_true, y_prob):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best_f1, best_thr = -1.0, 0.5
    for t in thr_grid:
        m = compute_metrics(y_true, y_prob, t)
        if m["F1"] > best_f1:
            best_f1 = m["F1"]
            best_thr = float(t)
    return best_thr


def summarize_f1_only(y_true, y_prob, model_name, feature_name):
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    thr = best_threshold_f1(y_true, y_prob)
    m = compute_metrics(y_true, y_prob, thr)
    return {
        "Model": model_name,
        "Feature": feature_name,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "Brier": brier,
        "Thr_F1": thr,
        "Accuracy@F1": m["Accuracy"],
        "Precision@F1": m["Precision/PPV"],
        "Recall@F1": m["Recall/Sensitivity"],
        "Specificity@F1": m["Specificity"],
        "F1@F1": m["F1"],
        "NPV@F1": m["NPV"],
        "Youden@F1": m["Youden"],
        "TN@F1": m["TN"],
        "FP@F1": m["FP"],
        "FN@F1": m["FN"],
        "TP@F1": m["TP"],
    }


def make_numeric_preprocess():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), [0])
    ], remainder="drop")


def make_tree_preprocess():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), [0])
    ], remainder="drop")


def get_base_model_specs() -> Dict[str, Tuple[callable, bool]]:
    """
    返回: 模型名 -> (factory函数, 是否用于 RF+GB 集成底座)
    """
    specs = {}

    def lr_factory():
        base = Pipeline([
            ("prep", make_numeric_preprocess()),
            ("clf", LogisticRegression(
                solver="lbfgs", penalty="l2", class_weight="balanced",
                max_iter=1000, random_state=SEED
            ))
        ])
        param_grid = {"clf__C": [0.1, 0.3, 1.0, 3.0, 10.0]}
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return GridSearchCV(base, param_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
    specs["Logistic Regression"] = (lr_factory, False)

    def svm_factory():
        base = Pipeline([
            ("prep", make_numeric_preprocess()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))
        ])
        param_grid = {"clf__C": [0.1, 0.3, 1, 3, 10], "clf__gamma": ["scale", 0.01, 0.03, 0.1]}
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return GridSearchCV(base, param_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
    specs["SVM"] = (svm_factory, False)

    def rf_factory():
        base = Pipeline([
            ("prep", make_tree_preprocess()),
            ("clf", RandomForestClassifier(
                n_estimators=700,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ))
        ])
        param_grid = {"clf__max_depth": [None, 6, 10], "clf__min_samples_leaf": [1, 3, 6]}
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return GridSearchCV(base, param_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
    specs["RandomForest"] = (rf_factory, True)

    def gb_factory():
        base = Pipeline([
            ("prep", make_tree_preprocess()),
            ("clf", GradientBoostingClassifier(random_state=SEED))
        ])
        param_grid = {
            "clf__learning_rate": [0.03, 0.1],
            "clf__n_estimators": [400, 700],
            "clf__max_depth": [2, 3],
        }
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return GridSearchCV(base, param_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
    specs["GradientBoosting"] = (gb_factory, True)

    if HAS_LGBM:
        def lgbm_factory():
            base = Pipeline([
                ("prep", make_tree_preprocess()),
                ("clf", LGBMClassifier(
                    n_estimators=900, learning_rate=0.05, objective="binary",
                    class_weight="balanced", random_state=SEED, verbosity=-1
                ))
            ])
            param_grid = {
                "clf__num_leaves": [31, 63],
                "clf__max_depth": [-1, 6],
                "clf__min_child_samples": [10, 20]
            }
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            return GridSearchCV(base, param_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
        specs["LightGBM"] = (lgbm_factory, False)

    if HAS_XGB:
        def xgb_factory():
            scale_pos = 1.0
            base = Pipeline([
                ("prep", make_tree_preprocess()),
                ("clf", XGBClassifier(
                    n_estimators=700,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos,
                ))
            ])
            param_grid = {
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__n_estimators": [400, 700],
            }
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            return GridSearchCV(base, param_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
        specs["XGBoost"] = (xgb_factory, False)

    if HAS_CATBOOST:
        def catboost_factory():
            base = Pipeline([
                ("prep", make_tree_preprocess()),
                ("clf", CatBoostClassifier(
                    depth=6,
                    learning_rate=0.1,
                    iterations=700,
                    loss_function="Logloss",
                    verbose=False,
                    random_state=SEED,
                    auto_class_weights="Balanced",
                    allow_writing_files=False,
                ))
            ])
            param_grid = {
                "clf__depth": [4, 6, 8],
                "clf__learning_rate": [0.03, 0.1],
                "clf__iterations": [500, 900],
            }
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            return GridSearchCV(base, param_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
        specs["CatBoost"] = (catboost_factory, False)

    def dnn_factory():
        base = Pipeline([
            ("prep", make_numeric_preprocess()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=1000,
                random_state=SEED,
            ))
        ])
        param_grid = {
            "clf__hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
            "clf__alpha": [1e-4, 1e-3],
        }
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        return GridSearchCV(base, param_grid, cv=inner, scoring="roc_auc", n_jobs=-1, refit=True)
    specs["DNN"] = (dnn_factory, False)

    return specs


def get_oof_proba(estimator_factory, X, y, calibrate=False, calibration_method="isotonic", calibration_cv=3):
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    fold_models = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]

        model = estimator_factory()
        model.fit(X_tr, y_tr)
        if calibrate:
            cal = CalibratedClassifierCV(model, method=calibration_method, cv=calibration_cv)
            cal.fit(X_tr, y_tr)
            proba = cal.predict_proba(X_va)[:, 1]
        else:
            proba = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba
        fold_models.append(model)
    return oof, fold_models


def fit_full_and_predict(estimator_factory, X_train, y_train, X_test, calibrate=False, calibration_method="isotonic", calibration_cv=3):
    model = estimator_factory()
    model.fit(X_train, y_train)
    if calibrate:
        cal = CalibratedClassifierCV(model, method=calibration_method, cv=calibration_cv)
        cal.fit(X_train, y_train)
        proba = cal.predict_proba(X_test)[:, 1]
        return model, cal, proba
    proba = model.predict_proba(X_test)[:, 1]
    return model, None, proba


def optimize_logit_convex_with_temperature(y_true, p_list, w_grid=None, T_grid=None):
    if w_grid is None:
        w_grid = np.linspace(0, 1, 101)
    if T_grid is None:
        T_grid = np.linspace(0.5, 2.0, 31)

    p1, p2 = p_list
    z1, z2 = safe_logit(p1), safe_logit(p2)
    mu1, sd1 = float(np.mean(z1)), float(np.std(z1) + 1e-8)
    mu2, sd2 = float(np.mean(z2)), float(np.std(z2) + 1e-8)
    z1 = (z1 - mu1) / sd1
    z2 = (z2 - mu2) / sd2

    best = {"loss": 1e9, "w": (0.5, 0.5), "T": 1.0}
    y = np.asarray(y_true).astype(float)
    for w in w_grid:
        z = w * z1 + (1 - w) * z2
        for T in T_grid:
            p = expit(z / T)
            loss = bce_loss(y, p)
            if loss < best["loss"]:
                best = {"loss": loss, "w": (float(w), float(1 - w)), "T": float(T)}
    return best["w"], best["T"], best["loss"]


def optimize_logit_convex_for_auc(y_true, p_list, w_grid=None, T_grid=None, metric="roc"):
    if w_grid is None:
        w_grid = np.linspace(0, 1, 101)
    if T_grid is None:
        T_grid = np.linspace(0.5, 2.0, 31)
    p1, p2 = p_list
    z1, z2 = safe_logit(p1), safe_logit(p2)
    y = np.asarray(y_true).astype(int)
    best = {"score": -1e9, "w": (0.5, 0.5), "T": 1.0}
    for w in w_grid:
        z = w * z1 + (1 - w) * z2
        for T in T_grid:
            p = expit(z / T)
            s = average_precision_score(y, p) if metric == "pr" else roc_auc_score(y, p)
            if s > best["score"]:
                best = {"score": s, "w": (float(w), float(1 - w)), "T": float(T)}
    return best["w"], best["T"], best["score"]


def rank_average_then_platt(y_true, p_list, weights=None):
    n = len(y_true)
    ranks = []
    for p in p_list:
        order = np.argsort(np.argsort(p))
        ranks.append((order + 1) / n)
    if weights is None:
        r_mean = np.mean(ranks, axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        r_mean = np.zeros_like(ranks[0], dtype=float)
        for wi, ri in zip(w, ranks):
            r_mean += wi * ri
    r_mean = r_mean.reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(r_mean, y_true)
    prob = lr.predict_proba(r_mean)[:, 1]
    return lr, prob


def to_rank_mean(p_list, weights=None):
    n = len(p_list[0])
    ranks = []
    for p in p_list:
        order = np.argsort(np.argsort(p))
        ranks.append((order + 1) / n)
    if weights is None:
        r = np.mean(ranks, axis=0)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        r = np.zeros_like(ranks[0], dtype=float)
        for wi, ri in zip(w, ranks):
            r += wi * ri
    return r.reshape(-1, 1)


def evaluate_single_feature_base_model(feature, model_name, factory, X_tr, y_tr, X_te, y_te):
    oof_prob, _ = get_oof_proba(factory, X_tr, y_tr, calibrate=False)
    thr = best_threshold_f1(y_tr, oof_prob)
    _, _, test_prob = fit_full_and_predict(factory, X_tr, y_tr, X_te, calibrate=False)
    row = summarize_f1_only(y_te, test_prob, model_name, feature)
    row["Thr_F1"] = thr
    test_metrics = compute_metrics(y_te, test_prob, thr)
    row.update({
        "Accuracy@F1": test_metrics["Accuracy"],
        "Precision@F1": test_metrics["Precision/PPV"],
        "Recall@F1": test_metrics["Recall/Sensitivity"],
        "Specificity@F1": test_metrics["Specificity"],
        "F1@F1": test_metrics["F1"],
        "NPV@F1": test_metrics["NPV"],
        "Youden@F1": test_metrics["Youden"],
        "TN@F1": test_metrics["TN"],
        "FP@F1": test_metrics["FP"],
        "FN@F1": test_metrics["FN"],
        "TP@F1": test_metrics["TP"],
    })
    return row, test_prob


def evaluate_single_feature_ensembles(feature, X_tr, y_tr, X_te, y_te, rf_factory, gb_factory):
    rows = []
    preds = []

    rf_oof, _ = get_oof_proba(rf_factory, X_tr, y_tr, calibrate=True, calibration_method="isotonic", calibration_cv=3)
    gb_oof, _ = get_oof_proba(gb_factory, X_tr, y_tr, calibrate=True, calibration_method="isotonic", calibration_cv=3)

    w_opt, T_opt, _ = optimize_logit_convex_with_temperature(y_tr.values, [rf_oof, gb_oof])
    w_auc, T_auc, _ = optimize_logit_convex_for_auc(y_tr.values, [rf_oof, gb_oof], metric="roc")

    _, _, rf_te = fit_full_and_predict(rf_factory, X_tr, y_tr, X_te, calibrate=True, calibration_method="isotonic", calibration_cv=3)
    _, _, gb_te = fit_full_and_predict(gb_factory, X_tr, y_tr, X_te, calibrate=True, calibration_method="isotonic", calibration_cv=3)

    # 1) HemoPain-Ensemble
    z_te = w_opt[0] * safe_logit(rf_te) + w_opt[1] * safe_logit(gb_te)
    p_blendT_te = expit(z_te / T_opt)
    z_oof = w_opt[0] * safe_logit(rf_oof) + w_opt[1] * safe_logit(gb_oof)
    p_blendT_oof = expit(z_oof / T_opt)
    thr = best_threshold_f1(y_tr, p_blendT_oof)
    row = summarize_f1_only(y_te, p_blendT_te, "HemoPain-Ensemble", feature)
    row["Thr_F1"] = thr
    m = compute_metrics(y_te, p_blendT_te, thr)
    for k, nk in [("Accuracy", "Accuracy@F1"), ("Precision/PPV", "Precision@F1"), ("Recall/Sensitivity", "Recall@F1"),
                  ("Specificity", "Specificity@F1"), ("F1", "F1@F1"), ("NPV", "NPV@F1"), ("Youden", "Youden@F1")]:
        row[nk] = m[k]
    row.update({"TN@F1": m["TN"], "FP@F1": m["FP"], "FN@F1": m["FN"], "TP@F1": m["TP"]})
    rows.append(row)
    preds.append(("HemoPain-Ensemble", p_blendT_te))

    # 2) ENS-LogitT-AUC
    z_te_auc = w_auc[0] * safe_logit(rf_te) + w_auc[1] * safe_logit(gb_te)
    p_blendT_auc_te = expit(z_te_auc / T_auc)
    z_oof_auc = w_auc[0] * safe_logit(rf_oof) + w_auc[1] * safe_logit(gb_oof)
    p_blendT_auc_oof = expit(z_oof_auc / T_auc)
    thr = best_threshold_f1(y_tr, p_blendT_auc_oof)
    row = summarize_f1_only(y_te, p_blendT_auc_te, "ENS-LogitT-AUC", feature)
    row["Thr_F1"] = thr
    m = compute_metrics(y_te, p_blendT_auc_te, thr)
    for k, nk in [("Accuracy", "Accuracy@F1"), ("Precision/PPV", "Precision@F1"), ("Recall/Sensitivity", "Recall@F1"),
                  ("Specificity", "Specificity@F1"), ("F1", "F1@F1"), ("NPV", "NPV@F1"), ("Youden", "Youden@F1")]:
        row[nk] = m[k]
    row.update({"TN@F1": m["TN"], "FP@F1": m["FP"], "FN@F1": m["FN"], "TP@F1": m["TP"]})
    rows.append(row)
    preds.append(("ENS-LogitT-AUC", p_blendT_auc_te))

    # 3) ENS-RankPlatt
    auc_w1 = roc_auc_score(y_tr, rf_oof)
    auc_w2 = roc_auc_score(y_tr, gb_oof)
    best_w = float(auc_w1 / (auc_w1 + auc_w2 + 1e-12))
    platt_lr, p_rank_oof = rank_average_then_platt(y_tr, [rf_oof, gb_oof], weights=[best_w, 1.0 - best_w])
    r_te = to_rank_mean([rf_te, gb_te], weights=[best_w, 1.0 - best_w])
    p_rank_te = platt_lr.predict_proba(r_te)[:, 1]
    thr = best_threshold_f1(y_tr, p_rank_oof)
    row = summarize_f1_only(y_te, p_rank_te, "ENS-RankPlatt", feature)
    row["Thr_F1"] = thr
    m = compute_metrics(y_te, p_rank_te, thr)
    for k, nk in [("Accuracy", "Accuracy@F1"), ("Precision/PPV", "Precision@F1"), ("Recall/Sensitivity", "Recall@F1"),
                  ("Specificity", "Specificity@F1"), ("F1", "F1@F1"), ("NPV", "NPV@F1"), ("Youden", "Youden@F1")]:
        row[nk] = m[k]
    row.update({"TN@F1": m["TN"], "FP@F1": m["FP"], "FN@F1": m["FN"], "TP@F1": m["TP"]})
    rows.append(row)
    preds.append(("ENS-RankPlatt", p_rank_te))

    # 4) ENS-Stack
    X_meta_oof = np.column_stack([safe_logit(rf_oof), safe_logit(gb_oof)])
    lr_meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=SEED)
    lr_meta.fit(X_meta_oof, y_tr)
    p_stack_oof = lr_meta.predict_proba(X_meta_oof)[:, 1]
    X_meta_te = np.column_stack([safe_logit(rf_te), safe_logit(gb_te)])
    p_stack_te = lr_meta.predict_proba(X_meta_te)[:, 1]
    thr = best_threshold_f1(y_tr, p_stack_oof)
    row = summarize_f1_only(y_te, p_stack_te, "ENS-Stack", feature)
    row["Thr_F1"] = thr
    m = compute_metrics(y_te, p_stack_te, thr)
    for k, nk in [("Accuracy", "Accuracy@F1"), ("Precision/PPV", "Precision@F1"), ("Recall/Sensitivity", "Recall@F1"),
                  ("Specificity", "Specificity@F1"), ("F1", "F1@F1"), ("NPV", "NPV@F1"), ("Youden", "Youden@F1")]:
        row[nk] = m[k]
    row.update({"TN@F1": m["TN"], "FP@F1": m["FP"], "FN@F1": m["FN"], "TP@F1": m["TP"]})
    rows.append(row)
    preds.append(("ENS-Stack", p_stack_te))

    return rows, preds


def main():
    ensure_dir(OUT_DIR)
    print("Running file:", os.path.abspath(__file__))
    print("Output dir:", OUT_DIR)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
        raise ValueError(f"未找到目标列 {TARGET_COL}")

    usable_features = [f for f in SINGLE_FEATURES if f in train_df.columns and f in test_df.columns]
    missing_features = [f for f in SINGLE_FEATURES if f not in usable_features]
    if missing_features:
        print("[Warning] 以下特征在 train/test 中缺失，将跳过:", missing_features)
    print("Usable single features:", usable_features)

    y_tr = train_df[TARGET_COL].astype(int)
    y_te = test_df[TARGET_COL].astype(int)
    print(f"Train: n={len(train_df)}, pos={int(y_tr.sum())}, pos_rate={y_tr.mean():.4f}")
    print(f"Test : n={len(test_df)}, pos={int(y_te.sum())}, pos_rate={y_te.mean():.4f}")

    model_specs = get_base_model_specs()
    print("Models:", list(model_specs.keys()) + ["HemoPain-Ensemble", "ENS-LogitT-AUC", "ENS-RankPlatt", "ENS-Stack"])

    result_rows = []
    pred_rows = []

    for feature in usable_features:
        print("\n" + "=" * 80)
        print(f"Single feature: {feature}")
        print("=" * 80)

        X_tr = train_df[[feature]].copy()
        X_te = test_df[[feature]].copy()

        # 基础模型
        for model_name, (factory, _) in model_specs.items():
            print(f"  -> Base model: {model_name}")
            try:
                row, test_prob = evaluate_single_feature_base_model(feature, model_name, factory, X_tr, y_tr, X_te, y_te)
                result_rows.append(row)
                for yt, yp in zip(y_te.values, test_prob):
                    pred_rows.append({
                        "Feature": feature,
                        "Model": model_name,
                        "y_true": int(yt),
                        "y_prob": float(yp),
                    })
            except Exception as e:
                print(f"     [Skip] {model_name} with feature {feature}: {e}")

        # 集成模型
        if "RandomForest" in model_specs and "GradientBoosting" in model_specs:
            print("  -> Ensemble models (GB+RF)")
            try:
                rf_factory = model_specs["RandomForest"][0]
                gb_factory = model_specs["GradientBoosting"][0]
                ens_rows, ens_preds = evaluate_single_feature_ensembles(feature, X_tr, y_tr, X_te, y_te, rf_factory, gb_factory)
                result_rows.extend(ens_rows)
                for model_name, probs in ens_preds:
                    for yt, yp in zip(y_te.values, probs):
                        pred_rows.append({
                            "Feature": feature,
                            "Model": model_name,
                            "y_true": int(yt),
                            "y_prob": float(yp),
                        })
            except Exception as e:
                print(f"     [Skip] ensembles with feature {feature}: {e}")

    result_df = pd.DataFrame(result_rows)
    pred_df = pd.DataFrame(pred_rows)

    # 排序：先按 Feature，再按 AUC/AP 升序（效果最好放最下面）
    if not result_df.empty:
        result_df = result_df.sort_values(by=["Feature", "ROC_AUC", "PR_AUC"], ascending=[True, True, True]).reset_index(drop=True)

    summary_csv = os.path.join(OUT_DIR, "single_feature_model_comparison_f1_test.csv")
    summary_xlsx = os.path.join(OUT_DIR, "single_feature_model_comparison_f1_test.xlsx")
    preds_csv = os.path.join(OUT_DIR, "single_feature_test_predictions_long.csv")

    result_df.to_csv(summary_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    try:
        result_df.to_excel(summary_xlsx, index=False)
    except Exception:
        pass
    pred_df.to_csv(preds_csv, index=False, encoding="utf-8-sig", float_format="%.6f")

    print("\n" + "=" * 80)
    print("Saved files:")
    print(summary_csv)
    print(summary_xlsx)
    print(preds_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()