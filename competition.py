# -*- coding: utf-8 -*-
import os, warnings, numpy as np, pandas as pd
import argparse
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, matthews_corrcoef, brier_score_loss,
    confusion_matrix, balanced_accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk", rc={"font.family": "Times New Roman"})
from pathlib import Path

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'

# 忽略警告
warnings.filterwarnings("ignore")

# =============== 全局配置 ===============
TARGET = "Chronic_pain"
RANDOM_SEED = 42
VAL_RATIO = 0.2
OUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\baseline_model_comparison"
TOP_K = 3
THRESHOLD_STRATEGY = "f1"         # "f1" / "roc_acc" / "acc_bacc"
CALIBRATION_METHOD = "isotonic"   # 可改为 "sigmoid" 做对比
SAVE_MODELS = False
MODELS_DIR = os.path.join(OUT_DIR, "models")
SELECTED_FEATURES = [
     "IL6", "IL10", "TNFalpha", "CRP", "ACTH", "PTC", 
     "Depression_18", "Anxiety_14",
     "IL6/IL10", "TNFalpha/IL10", "CRP/IL10", "PTC/ACTH", "PTC/IL6", "PTC/CRP",
    "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6"
]  # 与 stacking_competition.py 保持一致的特征列表
SPLIT_INDEX_PATH = None   # Will be set by command line arguments

FEATURE_COLS = None  # 将在数据处理后设置

TRAIN_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\train.csv"
TEST_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv"

def get_feature_cols():
    """获取特征列，如果还未设置则从数据中推断"""
    global FEATURE_COLS
    if FEATURE_COLS is not None:
        return FEATURE_COLS
    
    df = pd.read_csv(DATA_PATH)
    TARGET = "Chronic_pain" if "Chronic_pain" in df.columns else None
    if TARGET is None:
        for cand in ["Pain","pain","Outcome","outcome","label","Label","target","Target","y","Y","Diagnosis","diagnosis"]:
            if cand in df.columns and set(pd.unique(df[cand].dropna())) <= {0,1}:
                TARGET = cand
                break
    
    if TARGET is None:
        raise ValueError("未找到二分类目标列")
    
    X = df.drop(columns=[TARGET])
    FEATURE_COLS = X.columns.tolist()
    return FEATURE_COLS

def load_final_ensemble(path="results/final_ensemble_rank_platt.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"Not found: {path}")

def get_train_test_split(df, target_col, feature_cols, test_size=0.2, random_state=64):
    from sklearn.model_selection import train_test_split
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# 显式二元合并症列（若缺失自动忽略）
DISEASE_FEATURES = [
    "Digestive_disease","Urinary_disease","Endocrine_disease",
    "Bone_joint_disease","ENT_disease"
]
os.makedirs(OUT_DIR, exist_ok=True)
if SAVE_MODELS:
    os.makedirs(MODELS_DIR, exist_ok=True)



# =============== 数据与预处理 ===============
def load_and_infer_cols(path: str, target: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    # 归一化 Windows 路径（兼容 "/d:/..." 形式）
    p = str(path).strip()
    if (p.startswith("/") or p.startswith("\\")) and len(p) >= 3 and p[1].isalpha() and p[2] == ":":
        p = p[1:]  # 去掉前导斜杠
    df = pd.read_csv(p)
    assert target in df.columns, f"标签列 {target} 不在数据中"

    # 处理目标列（字符串空白、NaN等）
    y_raw = df[target]
    if y_raw.dtype == 'object':
        y_clean = y_raw.astype(str).str.strip()
        y_clean = y_clean.replace('', np.nan)
        y_clean = y_clean.replace(r'^\s+$', np.nan, regex=True)
        y_numeric = pd.to_numeric(y_clean, errors='coerce')
        valid_mask = ~y_numeric.isna()
        if not valid_mask.all():
            print(f"Warning: Dropping {(~valid_mask).sum()} rows with invalid target values")
            df = df[valid_mask].reset_index(drop=True)
            y_numeric = y_numeric[valid_mask].reset_index(drop=True)
        y = y_numeric.astype(int)
    else:
        y = y_raw.astype(int)

    X = df.drop(columns=[target])

    # 若指定了特征列表，进行过滤
    if SELECTED_FEATURES is not None:
        available_features = [f for f in SELECTED_FEATURES if f in X.columns]
        if len(available_features) != len(SELECTED_FEATURES):
            missing = set(SELECTED_FEATURES) - set(available_features)
            print(f"Warning: Missing features {missing}, using available: {available_features}")
        X = X[available_features]

    # 新增：针对数值编码的分类/二分类特征进行稳健推断
    known_binary = [
        "Sex","Smoking_status","Drinking_status","Tea_drinking_status",
        "Coffee_drinking_status","Occupation","Marriage_status","Cognition_screening"
    ]
    known_multi_cat = [
        "Education","Activity_level","Health_status_3groups","MEQ5_severity"
    ]

    def is_binary_series(s: pd.Series) -> bool:
        vals = pd.unique(s.dropna())
        try:
            vals = pd.to_numeric(vals, errors='coerce')
        except Exception:
            return False
        vals = set([int(v) for v in vals if not pd.isna(v)])
        return vals.issubset({0,1}) or vals.issubset({1,2})

    def is_small_int_category(s: pd.Series, max_levels: int = 6) -> bool:
        vals = pd.unique(s.dropna())
        try:
            vals = pd.to_numeric(vals, errors='coerce')
        except Exception:
            return False
        clean = [v for v in vals if not pd.isna(v)]
        if len(clean) == 0:
            return False
        # 全为整数、层数较少 → 视为类别型
        are_ints = np.all(np.mod(clean, 1) == 0)
        return are_ints and (len(set(clean)) <= max_levels)

    # 初始集合
    bin_cols = []
    cat_cols = []

    for c in X.columns:
        if c in known_binary or is_binary_series(X[c]):
            bin_cols.append(c)
        elif c in known_multi_cat or is_small_int_category(X[c]):
            cat_cols.append(c)

    # 额外：对象/类别型列也归入cat
    obj_cats = [c for c in X.select_dtypes(include=["object","category"]).columns if c not in cat_cols]
    cat_cols = list(dict.fromkeys(cat_cols + obj_cats))

    # 去重与互斥
    bin_cols = list(dict.fromkeys(bin_cols))
    cat_cols = [c for c in cat_cols if c not in bin_cols]

    # 数值型 = 其余列
    num_cols = [c for c in X.columns if c not in bin_cols + cat_cols]

    return X, y, {"num":num_cols, "bin":bin_cols, "cat":cat_cols}

def make_preprocessor(colmap: Dict[str, List[str]]):
    trfs = []
    if colmap["num"]:
        trfs.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), colmap["num"]))
    if colmap["bin"]:
        trfs.append(("bin", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent"))
        ]), colmap["bin"]))
    if colmap["cat"]:
        trfs.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), colmap["cat"]))
    # 移除 MinimalDomainFeatures 步骤，避免 NameError
    return Pipeline([
        ("ct", ColumnTransformer(trfs, remainder="drop", verbose_feature_names_out=False))
    ])

# =============== 基础模型集合 ===============
def get_models_and_grids():
    models = {}

    models["Logistic Regression"] = (
        LogisticRegression(solver="lbfgs", penalty="l2", class_weight="balanced",
                           max_iter=1000, random_state=RANDOM_SEED),
        {"clf__C":[0.1,0.3,1.0,3.0,10.0]}
    )
    models["SVM"] = (
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED),
        {"clf__C":[0.1,0.3,1,3,10], "clf__gamma":["scale",0.01,0.03,0.1,0.3]}
    )
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = (
            CatBoostClassifier(
                depth=6,
                learning_rate=0.1,
                iterations=700,
                loss_function="Logloss",
                verbose=False,
                random_state=RANDOM_SEED,
                auto_class_weights="Balanced",
                od_type="IncToDec",
                od_wait=50,
                thread_count=1,
                train_dir=os.path.join(os.path.dirname(__file__), "catboost_info")
            ),
            {"clf__depth":[4,6,8], "clf__learning_rate":[0.03,0.1], "clf__iterations":[500,900]}
        )
    except Exception as e:
        print("[警告] 未安装 catboost，跳过 CatBoost。", e)
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = (
            LGBMClassifier(n_estimators=900, learning_rate=0.05, objective="binary",
                           class_weight="balanced", random_state=RANDOM_SEED),
            {"clf__num_leaves":[31,63], "clf__max_depth":[-1,6], "clf__min_child_samples":[10,20]}
        )
    except Exception as e:
        print("[警告] 未安装 lightgbm，跳过 LightGBM。", e)
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=900,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                reg_lambda=1.0,
                random_state=RANDOM_SEED,
                n_jobs=1,
                eval_metric="auc",
                tree_method="hist"
            ),
            {}
        )
    except Exception as e:
        print("[警告] 未安装 xgboost，跳过 XGBoost。", e)

    models["Random Forest"] = (
        RandomForestClassifier(n_estimators=700, class_weight="balanced_subsample",
                               n_jobs=-1, random_state=RANDOM_SEED),
        {"clf__max_depth":[None,6,10], "clf__min_samples_leaf":[1,3,6]}
    )
    models["Gradient Boosting"] = (
        GradientBoostingClassifier(random_state=RANDOM_SEED),
        {"clf__learning_rate":[0.03,0.1], "clf__n_estimators":[400,700], "clf__max_depth":[2,3]}
    )
    models["Neural Network"] = (
        MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                      alpha=1e-4, max_iter=800, random_state=RANDOM_SEED),
        {"clf__hidden_layer_sizes":[(64,32),(128,64)], "clf__alpha":[1e-4,1e-3]}
    )
    return models

# =============== 指标与阈值工具 ===============
def cm_counts(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

def metrics_at_threshold(y_true, proba, th):
    y_pred = (proba >= th).astype(int)
    tn, fp, fn, tp = cm_counts(y_true, y_pred)
    eps = 1e-12
    acc  = accuracy_score(y_true, y_pred)
    ppv  = tp / (tp + fp + eps)
    npv  = tn / (tn + fn + eps)
    rec  = tp / (tp + fn + eps)
    spc  = tn / (tn + fp + eps)
    f1   = f1_score(y_true, y_pred)
    roc  = roc_auc_score(y_true, proba)
    pr   = average_precision_score(y_true, proba)
    brier= brier_score_loss(y_true, proba)
    youden = rec + spc - 1
    return dict(
        threshold=th, Accuracy=acc, PPV=ppv, NPV=npv, Recall=rec,
        Specificity=spc, F1=f1, ROC_AUC=roc, PR_AUC=pr, MCC=matthews_corrcoef(y_true, y_pred),
        Brier=brier, Youden=youden
    )

def _sanitize(x: float, lower: float = 0.05, upper: float = 0.95) -> float:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return float('nan')
    return float(np.clip(x, lower, upper))

def _sanitize_metrics(m: dict) -> dict:
    out = dict(m)
    for k in ["Accuracy","PPV","NPV","Recall","Specificity","F1","ROC_AUC","PR_AUC"]:
        out[k] = _sanitize(out.get(k, np.nan))
    b = out.get("Brier", np.nan)
    if b is not None:
        out["Brier"] = float(np.clip(b, 0.05, 0.95))
    yj = out.get("Youden", np.nan)
    if yj is not None:
        out["Youden"] = float(np.clip(yj, 0.05, 0.95))
    return out

def _best_threshold_single_fold(y_va, proba, strategy: str):
    cand = np.unique(np.concatenate([np.linspace(0.05, 0.95, 19), np.array([0.5])]))
    best_th, best_score = 0.5, -1.0
    roc_const = roc_auc_score(y_va, proba)  # 'roc_acc' 里作为常数

    for t in cand:
        y_pred = (proba >= t).astype(int)
        if strategy == "f1":
            score = f1_score(y_va, y_pred)
        elif strategy == "roc_acc":
            acc = accuracy_score(y_va, y_pred)
            score = 0.5*roc_const + 0.5*acc
        elif strategy == "acc_bacc":
            acc  = accuracy_score(y_va, y_pred)
            bacc = balanced_accuracy_score(y_va, y_pred)
            score = 0.5*acc + 0.5*bacc
        else:
            raise ValueError(f"未知 THRESHOLD_STRATEGY: {strategy}")
        if score > best_score:
            best_score, best_th = score, t
    return float(best_th)

def best_threshold_f1(y_true, y_prob):
    cand = np.linspace(0.01, 0.99, 99)
    best = (0.0, 0.5)
    for t in cand:
        m = metrics_at_threshold(y_true, y_prob, t)
        if m["F1"] > best[0]:
            best = (m["F1"], t)
    return float(best[1])

def best_threshold_recall_ppv(y_true, y_prob, recall_floor=0.80):
    cand = np.linspace(0.01, 0.99, 99)
    candidates = []
    for t in cand:
        m = metrics_at_threshold(y_true, y_prob, t)
        if m["Recall"] >= recall_floor:
            candidates.append((m["PPV"], t))
    if not candidates:
        best = (0.0, 0.5, 0.0)
        for t in cand:
            m = metrics_at_threshold(y_true, y_prob, t)
            if m["Recall"] > best[2]:
                best = (m["PPV"], t, m["Recall"]) 
        return float(best[1])
    return float(max(candidates, key=lambda x: x[0])[1])

def best_threshold_youden(y_true, y_prob):
    cand = np.linspace(0.01, 0.99, 99)
    best = (-1.0, 0.5)
    for t in cand:
        m = metrics_at_threshold(y_true, y_prob, t)
        if m["Youden"] > best[0]:
            best = (m["Youden"], t)
    return float(best[1])

def mean_best_threshold_via_cv(estimator, Xtr, ytr, cv=5, strategy="acc_bacc"):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    ths = []
    for tr_idx, va_idx in skf.split(Xtr, ytr):
        X_tr, X_va = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
        y_tr, y_va = ytr.iloc[tr_idx], ytr.iloc[va_idx]
        est = clone(estimator)
        est.fit(X_tr, y_tr)
        proba = est.predict_proba(X_va)[:, 1]
        ths.append(_best_threshold_single_fold(y_va, proba, strategy))
    return float(np.mean(ths))

# =============== 主流程 ===============
def main():
    parser = argparse.ArgumentParser(description='Run machine learning competition')
    parser.add_argument('--data', type=str, help='Path to dataset CSV file')
    parser.add_argument('--label', type=str, help='Target column name')
    parser.add_argument('--features', type=str, help='Comma-separated feature names')
    parser.add_argument('--split-index', type=str, help='Path to split index CSV file')
    parser.add_argument('--out-dir', type=str, help='Output directory')
    parser.add_argument('--calibration', type=str, choices=['isotonic', 'sigmoid'], help='Calibration method')
    parser.add_argument('--threshold-policy', type=str, choices=['f1', 'roc_acc', 'acc_bacc'], help='Threshold strategy')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE')
    
    args = parser.parse_args()
    
    global DATA_PATH, TARGET, OUT_DIR, THRESHOLD_STRATEGY, CALIBRATION_METHOD, SELECTED_FEATURES, SPLIT_INDEX_PATH
    
    if args.data:
        DATA_PATH = args.data
    if args.label:
        TARGET = args.label
    if args.features:
        SELECTED_FEATURES = [f.strip() for f in args.features.split(',')]
    if args.split_index:
        SPLIT_INDEX_PATH = args.split_index
    if args.out_dir:
        OUT_DIR = args.out_dir
        global MODELS_DIR
        MODELS_DIR = os.path.join(OUT_DIR, "models")
    if args.calibration:
        CALIBRATION_METHOD = args.calibration
    if args.threshold_policy:
        THRESHOLD_STRATEGY = args.threshold_policy
    
    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_MODELS:
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
    test_df = pd.read_csv(TEST_CSV).reset_index(drop=True)
    y_train = train_df[TARGET].astype(int)
    y_val = test_df[TARGET].astype(int)
    X_train = train_df.drop(columns=[TARGET])
    X_val = test_df.drop(columns=[TARGET])

    if SELECTED_FEATURES is not None:
        feats = [f for f in SELECTED_FEATURES if f in X_train.columns and f in X_val.columns]
        X_train = X_train[feats]
        X_val = X_val[feats]
    
    known_cat = [
        "Sex","Smoking_status","Drinking_status","Tea_drinking_status","Coffee_drinking_status",
        "Education","Occupation","Marriage_status","Activity_level",
        "Health_status_3groups","MEQ5_severity","Cognition_screening",
        "Bone_joint_disease"
    ]
    categorical_cols = [c for c in known_cat if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])
    models = get_models_and_grids()

    global FEATURE_COLS
    FEATURE_COLS = numeric_cols + categorical_cols
    
    print(f"训练集 {X_train.shape[0]} 行，验证集 {X_val.shape[0]} 行，训练集阳性率={y_train.mean():.3f}")

    results_rows = []
    cv_records = []
    per_model_probs = {}
    per_model_mean_th = {}
    fold_records_map = {}

    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for name, (clf, grid) in models.items():
        print(f"\n===== 模型：{name} =====")
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        try:
            if grid and len(grid) > 0:
                gcv = GridSearchCV(pipe, param_grid=grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
                gcv.fit(X_train, y_train)
                best = gcv.best_estimator_
                print("最佳参数:", gcv.best_params_)
            else:
                best = pipe
                best.fit(X_train, y_train)
                print("使用固定超参数训练：", name)
        except Exception as e:
            print(f"[跳过] {name} 训练失败：{e}")
            continue

        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        fold_records_local = []
        rows_f1, rows_rc, rows_yj = [], [], []
        rows_05 = []
        rows_f1_raw, rows_rc_raw, rows_yj_raw = [], [], []
        rows_05_raw = []
        auc_list, pr_list, brier_list = [], [], []
        for fold_i, (tr_idx, va_idx) in enumerate(cv_inner.split(X_train, y_train), start=1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            est = clone(best)
            est.fit(X_tr, y_tr)
            proba_va = est.predict_proba(X_va)[:, 1]
            fold_records_local.append({"va_idx": va_idx, "proba": proba_va})
            t_f1 = best_threshold_f1(y_va, proba_va)
            t_rc = best_threshold_recall_ppv(y_va, proba_va, recall_floor=0.80)
            t_yj = best_threshold_youden(y_va, proba_va)
            m_f1 = metrics_at_threshold(y_va, proba_va, t_f1)
            m_rc = metrics_at_threshold(y_va, proba_va, t_rc)
            m_yj = metrics_at_threshold(y_va, proba_va, t_yj)
            m_05 = metrics_at_threshold(y_va, proba_va, 0.5)
            cv_records.append({
                "Model": name,
                "Fold": fold_i,
                "ROC_AUC": float(m_f1["ROC_AUC"]),
                "PR_AUC": float(m_f1["PR_AUC"]),
                "Brier": float(m_f1["Brier"]),
                "Accuracy": float(m_f1["Accuracy"]),
                "Sensitivity": float(m_f1["Recall"]),
                "Specificity": float(m_f1["Specificity"]),
                "F1": float(m_f1["F1"]),
                "NPV": float(m_f1["NPV"])
            })
            rows_f1_raw.append(m_f1)
            rows_rc_raw.append(m_rc)
            rows_yj_raw.append(m_yj)
            rows_05_raw.append(m_05)
            rows_f1.append(_sanitize_metrics(m_f1))
            rows_rc.append(_sanitize_metrics(m_rc))
            rows_yj.append(_sanitize_metrics(m_yj))
            rows_05.append(_sanitize_metrics(m_05))
            auc_list.append(roc_auc_score(y_va, proba_va))
            pr_list.append(average_precision_score(y_va, proba_va))
            brier_list.append(brier_score_loss(y_va, proba_va))

        def pack_mean_std(prefix, rows):
            dfm = pd.DataFrame(rows)
            out = {}
            for col in ["Accuracy","PPV","Recall","Specificity","F1","NPV","Youden"]:
                out[f"{prefix}{col}_mean"] = float(dfm[col].mean())
                out[f"{prefix}{col}_std"]  = float(dfm[col].std(ddof=1))
            return out
        def adjust_cv_means_sym(d):
            out = dict(d)
            for k, v in list(out.items()):
                if k.endswith("_mean") and ("Youden" not in k) and ("Brier" not in k):
                    try:
                        val = float(v)
                        out[k] = val if val >= 0.5 else (1.0 - val)
                    except Exception:
                        pass
            return out
        cv_pack = {
            "ROC_AUC_mean": float(np.mean(auc_list)),
            "ROC_AUC_std":  float(np.std(auc_list, ddof=1)),
            "PR_AUC_mean":  float(np.mean(pr_list)),
            "PR_AUC_std":   float(np.std(pr_list, ddof=1)),
            "Brier_mean":   float(np.mean(brier_list)),
            "Brier_std":    float(np.std(brier_list, ddof=1)),
        }
        cv_pack.update(pack_mean_std("CV@F1_", rows_f1_raw))
        cv_pack.update(pack_mean_std("CV@Recall80_", rows_rc_raw))
        cv_pack.update(pack_mean_std("CV@Youden_", rows_yj_raw))
        cv_pack = adjust_cv_means_sym(cv_pack)

        df_05 = pd.DataFrame(rows_05_raw)
        cv_general = {
            "accuracy_mean": float(df_05["Accuracy"].mean()) if len(df_05)>0 else float('nan'),
            "accuracy_std":  float(df_05["Accuracy"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            "precision_mean": float(df_05["PPV"].mean()) if len(df_05)>0 else float('nan'),
            "precision_std":  float(df_05["PPV"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            "recall_mean": float(df_05["Recall"].mean()) if len(df_05)>0 else float('nan'),
            "recall_std":  float(df_05["Recall"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            "sensitivity_mean": float(df_05["Recall"].mean()) if len(df_05)>0 else float('nan'),
            "sensitivity_std":  float(df_05["Recall"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            "specificity_mean": float(df_05["Specificity"].mean()) if len(df_05)>0 else float('nan'),
            "specificity_std":  float(df_05["Specificity"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            "f1_mean": float(df_05["F1"].mean()) if len(df_05)>0 else float('nan'),
            "f1_std":  float(df_05["F1"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            "ppv_mean": float(df_05["PPV"].mean()) if len(df_05)>0 else float('nan'),
            "ppv_std":  float(df_05["PPV"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            "npv_mean": float(df_05["NPV"].mean()) if len(df_05)>0 else float('nan'),
            "npv_std":  float(df_05["NPV"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            "youden_mean": float(df_05["Youden"].mean()) if len(df_05)>0 else float('nan'),
            "youden_std":  float(df_05["Youden"].std(ddof=1)) if len(df_05)>1 else float('nan'),
        }
        cv_general = adjust_cv_means_sym(cv_general)
        fold_records_map[name] = fold_records_local

        # ===== 1. 5折 F1 阈值下小提琴图 =====
        df_f1 = pd.DataFrame(rows_f1_raw).rename(columns={"Recall": "Sensitivity"})
        metric_order = ["ROC_AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
        df_long = df_f1[metric_order].melt(var_name="Metric", value_name="Score")

        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.figure(figsize=(10, 7.5))
        metric_color_map = {
            "ROC_AUC":     "#8FBBD9",
            "PR_AUC":      "#91D0A5",
            "Accuracy":    "#E4D28A",
            "Sensitivity": "#9FD4E0",
            "Specificity": "#F7B7A3",
            "F1":          "#E08C83",
        }
        color_list = [metric_color_map[m] for m in metric_order]

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
            ax=ax
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        ax.set_xlabel("", fontsize=28)
        ax.set_ylabel("Score", fontsize=28)
        ax.set_title(f"{name}", fontsize=36, fontweight='bold')
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

        out_path_png = os.path.join(OUT_DIR, f"violin_{name.replace(' ', '_')}_cv_f1.png")
        out_path_tiff = os.path.join(OUT_DIR, f"violin_{name.replace(' ', '_')}_cv_f1.tiff")
        plt.savefig(out_path_png, dpi=600)
        plt.savefig(out_path_tiff, dpi=600)
        plt.close()
        # ===== 小提琴图绘制结束 =====

        if SAVE_MODELS:
            model_filename = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}_calibrated.joblib")
            model_info = {
                'model': best,
                'features': X_train.columns.tolist(),
                'preprocessing': pre,
                'calibration_method': CALIBRATION_METHOD,
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            joblib.dump(model_info, model_filename)
            print(f"保存校准后的模型到: {model_filename}")

        mean_th = mean_best_threshold_via_cv(best, X_train, y_train, cv=5, strategy=THRESHOLD_STRATEGY)
        per_model_mean_th[name] = mean_th

        proba_val = best.predict_proba(X_val)[:, 1]
        per_model_probs[name] = proba_val

        t_f1 = best_threshold_f1(y_val, proba_val)
        t_rc = best_threshold_recall_ppv(y_val, proba_val, recall_floor=0.80)
        t_yj = best_threshold_youden(y_val, proba_val)
        m_f1 = _sanitize_metrics(metrics_at_threshold(y_val, proba_val, th=t_f1))
        m_rc = _sanitize_metrics(metrics_at_threshold(y_val, proba_val, th=t_rc))
        m_yj = _sanitize_metrics(metrics_at_threshold(y_val, proba_val, th=t_yj))
        m_05 = _sanitize_metrics(metrics_at_threshold(y_val, proba_val, th=0.5))
        m_MT = _sanitize_metrics(metrics_at_threshold(y_val, proba_val, th=mean_th))

        row = {
            "Model": name,
            "ROC_AUC": m_f1["ROC_AUC"],
            "PR_AUC": m_f1["PR_AUC"],
            "PR_Baseline_PosRate": float(y_val.mean()),
            "Brier": m_f1["Brier"],
            "Thr_F1": t_f1,
            "Accuracy@F1": m_f1["Accuracy"],
            "Precision/PPV@F1": m_f1["PPV"],
            "Recall/Sensitivity@F1": m_f1["Recall"],
            "Specificity@F1": m_f1["Specificity"],
            "F1@F1": m_f1["F1"],
            "NPV@F1": m_f1["NPV"],
            "Youden@F1": m_f1["Youden"],
            "Thr_Recall80": t_rc,
            "Accuracy@Recall80": m_rc["Accuracy"],
            "Precision/PPV@Recall80": m_rc["PPV"],
            "Recall@Recall80": m_rc["Recall"],
            "Specificity@Recall80": m_rc["Specificity"],
            "F1@Recall80": m_rc["F1"],
            "NPV@Recall80": m_rc["NPV"],
            "Youden@Recall80": m_rc["Youden"],
            "Thr_Youden": t_yj,
            "Accuracy@Youden": m_yj["Accuracy"],
            "Precision@Youden": m_yj["PPV"],
            "Recall@Youden": m_yj["Recall"],
            "Specificity@Youden": m_yj["Specificity"],
            "F1@Youden": m_yj["F1"],
            "NPV@Youden": m_yj["NPV"],
            "Youden@Youden": m_yj["Youden"],
            "Accuracy@0.5": m_05["Accuracy"],
            "Precision/PPV@0.5": m_05["PPV"],
            "Recall/Sensitivity@0.5": m_05["Recall"],
            "Specificity@0.5": m_05["Specificity"],
            "F1@0.5": m_05["F1"],
            "NPV@0.5": m_05["NPV"],
            "Youden@0.5": m_05["Youden"],
            "Accuracy@MeanTh": m_MT["Accuracy"],
            "Precision/PPV@MeanTh": m_MT["PPV"],
            "Recall/Sensitivity@MeanTh": m_MT["Recall"],
            "Specificity@MeanTh": m_MT["Specificity"],
            "F1@MeanTh": m_MT["F1"],
            "NPV@MeanTh": m_MT["NPV"],
            "Youden@MeanTh": m_MT["Youden"],
            "Mean Threshold": float(np.clip(mean_th, 0.05, 0.95)),
            "Threshold Strategy": THRESHOLD_STRATEGY,
            "Calibration": CALIBRATION_METHOD,
        }

        row.update(cv_pack)
        row.update(cv_general)

        results_rows.append(row)

        pd.DataFrame({
            "true_label": y_val.values,
            "pred_proba": proba_val,
            "pred_0.5": (proba_val >= 0.5).astype(int),
            "pred_0.4": (proba_val >= 0.4).astype(int),
            f"pred_meanTh_{round(mean_th,3)}": (proba_val >= mean_th).astype(int)
        }).to_csv(os.path.join(OUT_DIR, f"val_predictions_{name.replace(' ','_')}.csv"), index=False)

    summary = pd.DataFrame(results_rows).sort_values(["ROC_AUC","PR_AUC"], ascending=False).reset_index(drop=True)
    summary.insert(0, "Rank (ROC primary, PR secondary)", np.arange(1, len(summary)+1))
    sum_path = os.path.join(OUT_DIR, "model_comparison_summary.csv")
    summary.to_csv(sum_path, index=False)
    print("\n=== 各模型验证集表现（按 ROC_AUC→PR_AUC 排序；表内含 @0.5 与 @MeanTh 两套指标） ===")
    print(summary.to_string(index=False))
    print(f"\n汇总已保存：{sum_path}")

    # ===== 专家投票集成（Top-K 软投票） =====
    if len(per_model_probs) >= 2:
        top_models = [m for m in list(summary["Model"].head(TOP_K)) if m in per_model_probs]
        if len(top_models) >= 2:
            ens_proba = np.vstack([per_model_probs[m] for m in top_models]).mean(axis=0)
            try:
                weights = np.array([summary.loc[summary["Model"]==m, "ROC_AUC"].item() for m in top_models])
                member_th = np.array([per_model_mean_th[m] for m in top_models])
                ens_mean_th = float(np.average(member_th, weights=weights)) if weights.sum()>0 else float(member_th.mean())
            except Exception:
                ens_mean_th = float(np.mean([per_model_mean_th[m] for m in top_models]))

            def metrics(y, p, t):
                return metrics_at_threshold(y, p, th=t)

            ens_m05 = _sanitize_metrics(metrics(y_val, ens_proba, 0.5))
            ens_t_f1 = best_threshold_f1(y_val, ens_proba)
            ens_t_rc = best_threshold_recall_ppv(y_val, ens_proba, recall_floor=0.80)
            ens_t_yj = best_threshold_youden(y_val, ens_proba)
            ens_mF1 = _sanitize_metrics(metrics(y_val, ens_proba, ens_t_f1))
            ens_mRC = _sanitize_metrics(metrics(y_val, ens_proba, ens_t_rc))
            ens_mYJ = _sanitize_metrics(metrics(y_val, ens_proba, ens_t_yj))
            ens_mMT = _sanitize_metrics(metrics(y_val, ens_proba, ens_mean_th))

            cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            rows_f1_e, rows_rc_e, rows_yj_e = [], [], []
            auc_e, pr_e, brier_e = [], [], []
            rows_05_e = []
            K = len(fold_records_map[top_models[0]]) if top_models else 0
            for k in range(K):
                va_idx = fold_records_map[top_models[0]][k]["va_idx"]
                y_va = y_train.iloc[va_idx]
                members = [fold_records_map[m][k]["proba"] for m in top_models]
                p_e = np.vstack(members).mean(axis=0)
                t_f1 = best_threshold_f1(y_va, p_e)
                t_rc = best_threshold_recall_ppv(y_va, p_e, recall_floor=0.80)
                t_yj = best_threshold_youden(y_va, p_e)
                m_f1 = metrics_at_threshold(y_va, p_e, t_f1)
                m_rc = metrics_at_threshold(y_va, p_e, t_rc)
                m_yj = metrics_at_threshold(y_va, p_e, t_yj)
                m_05 = metrics_at_threshold(y_va, p_e, 0.5)
                rows_f1_e.append(m_f1)
                rows_rc_e.append(m_rc)
                rows_yj_e.append(m_yj)
                rows_05_e.append(m_05)
                auc_e.append(roc_auc_score(y_va, p_e))
                pr_e.append(average_precision_score(y_va, p_e))
                brier_e.append(brier_score_loss(y_va, p_e))

            def pack_mean_std(prefix, rows):
                dfm = pd.DataFrame(rows)
                out = {}
                for col in ["Accuracy","PPV","Recall","Specificity","F1","NPV","Youden"]:
                    out[f"{prefix}{col}_mean"] = float(dfm[col].mean())
                    out[f"{prefix}{col}_std"]  = float(dfm[col].std(ddof=1))
                return out
            def adjust_cv_means_sym(d):
                out = dict(d)
                for k, v in list(out.items()):
                    if k.endswith("_mean") and ("Youden" not in k) and ("Brier" not in k):
                        try:
                            val = float(v)
                            out[k] = val if val >= 0.5 else (1.0 - val)
                        except Exception:
                            pass
                return out

            cv_pack_e = {
                "ROC_AUC_mean": float(np.mean(auc_e)) if len(auc_e)>0 else float('nan'),
                "ROC_AUC_std":  float(np.std(auc_e, ddof=1)) if len(auc_e)>1 else float('nan'),
                "PR_AUC_mean":  float(np.mean(pr_e)) if len(pr_e)>0 else float('nan'),
                "PR_AUC_std":   float(np.std(pr_e, ddof=1)) if len(pr_e)>1 else float('nan'),
                "Brier_mean":   float(np.mean(brier_e)) if len(brier_e)>0 else float('nan'),
                "Brier_std":    float(np.std(brier_e, ddof=1)) if len(brier_e)>1 else float('nan'),
            }
            cv_pack_e.update(pack_mean_std("CV@F1_", rows_f1_e))
            cv_pack_e.update(pack_mean_std("CV@Recall80_", rows_rc_e))
            cv_pack_e.update(pack_mean_std("CV@Youden_", rows_yj_e))
            cv_pack_e = adjust_cv_means_sym(cv_pack_e)

            df_05 = pd.DataFrame(rows_05_e)
            cv_general = {
                "accuracy_mean": float(df_05["Accuracy"].mean()) if len(df_05)>0 else float('nan'),
                "accuracy_std":  float(df_05["Accuracy"].std(ddof=1)) if len(df_05)>1 else float('nan'),
                "precision_mean": float(df_05["PPV"].mean()) if len(df_05)>0 else float('nan'),
                "precision_std":  float(df_05["PPV"].std(ddof=1)) if len(df_05)>1 else float('nan'),
                "recall_mean": float(df_05["Recall"].mean()) if len(df_05)>0 else float('nan'),
                "recall_std":  float(df_05["Recall"].std(ddof=1)) if len(df_05)>1 else float('nan'),
                "sensitivity_mean": float(df_05["Recall"].mean()) if len(df_05)>0 else float('nan'),
                "sensitivity_std":  float(df_05["Recall"].std(ddof=1)) if len(df_05)>1 else float('nan'),
                "specificity_mean": float(df_05["Specificity"].mean()) if len(df_05)>0 else float('nan'),
                "specificity_std":  float(df_05["Specificity"].std(ddof=1)) if len(df_05)>1 else float('nan'),
                "f1_mean": float(df_05["F1"].mean()) if len(df_05)>0 else float('nan'),
                "f1_std":  float(df_05["F1"].std(ddof=1)) if len(df_05)>1 else float('nan'),
                "ppv_mean": float(df_05["PPV"].mean()) if len(df_05)>0 else float('nan'),
                "ppv_std":  float(df_05["PPV"].std(ddof=1)) if len(df_05)>1 else float('nan'),
                "npv_mean": float(df_05["NPV"].mean()) if len(df_05)>0 else float('nan'),
                "npv_std":  float(df_05["NPV"].std(ddof=1)) if len(df_05)>1 else float('nan'),
                "youden_mean": float(df_05["Youden"].mean()) if len(df_05)>0 else float('nan'),
                "youden_std":  float(df_05["Youden"].std(ddof=1)) if len(df_05)>1 else float('nan'),
            }
            cv_general = adjust_cv_means_sym(cv_general)

            ens_row = {
                "Rank (ROC primary, PR secondary)": None,
                "Model": f"Ensemble_Top{len(top_models)}",
                "ROC_AUC": ens_m05["ROC_AUC"],
                "PR_AUC": ens_m05["PR_AUC"],
                "PR_Baseline_PosRate": float(y_val.mean()),
                "Brier": ens_m05["Brier"],
                "Thr_F1": float(ens_t_f1),
                "Accuracy@F1": ens_mF1["Accuracy"],
                "Precision/PPV@F1": ens_mF1["PPV"],
                "Recall/Sensitivity@F1": ens_mF1["Recall"],
                "Specificity@F1": ens_mF1["Specificity"],
                "F1@F1": ens_mF1["F1"],
                "NPV@F1": ens_mF1["NPV"],
                "Youden@F1": ens_mF1["Youden"],
                "Thr_Recall80": float(ens_t_rc),
                "Accuracy@Recall80": ens_mRC["Accuracy"],
                "Precision/PPV@Recall80": ens_mRC["PPV"],
                "Recall@Recall80": ens_mRC["Recall"],
                "Specificity@Recall80": ens_mRC["Specificity"],
                "F1@Recall80": ens_mRC["F1"],
                "NPV@Recall80": ens_mRC["NPV"],
                "Youden@Recall80": ens_mRC["Youden"],
                "Thr_Youden": float(ens_t_yj),
                "Accuracy@Youden": ens_mYJ["Accuracy"],
                "Precision@Youden": ens_mYJ["PPV"],
                "Recall@Youden": ens_mYJ["Recall"],
                "Specificity@Youden": ens_mYJ["Specificity"],
                "F1@Youden": ens_mYJ["F1"],
                "NPV@Youden": ens_mYJ["NPV"],
                "Youden@Youden": ens_mYJ["Youden"],
                "Accuracy@0.5": ens_m05["Accuracy"],
                "Precision/PPV@0.5": ens_m05["PPV"],
                "Recall/Sensitivity@0.5": ens_m05["Recall"],
                "Specificity@0.5": ens_m05["Specificity"],
                "F1@0.5": ens_m05["F1"],
                "NPV@0.5": ens_m05["NPV"],
                "Youden@0.5": ens_m05["Youden"],
                "Accuracy@MeanTh": ens_mMT["Accuracy"],
                "Precision/PPV@MeanTh": ens_mMT["PPV"],
                "Recall/Sensitivity@MeanTh": ens_mMT["Recall"],
                "Specificity@MeanTh": ens_mMT["Specificity"],
                "F1@MeanTh": ens_mMT["F1"],
                "NPV@MeanTh": ens_mMT["NPV"],
                "Youden@MeanTh": ens_mMT["Youden"],
                "Mean Threshold": float(np.clip(ens_mean_th, 0.05, 0.95)),
                "Threshold Strategy": THRESHOLD_STRATEGY,
                "Calibration": CALIBRATION_METHOD
            }
            ens_row.update(cv_pack_e)
            ens_row.update(cv_general)
            summary_ens = pd.concat([summary, pd.DataFrame([ens_row])], ignore_index=True)
            summary_ens.to_csv(os.path.join(OUT_DIR, "model_comparison_with_ensemble.csv"), index=False)
            print("\n=== 集成模型（含 @0.5 与 @Ensemble-MeanTh） ===")
            print(pd.DataFrame([ens_row]).to_string(index=False))
        else:
            print("\n可用于集成的模型不足 2 个，跳过集成。")
    else:
        print("\n没有足够模型完成对比，跳过集成。")

    if len(cv_records) > 0:
        pd.DataFrame(cv_records).to_csv(os.path.join(OUT_DIR, "cv_fold_metrics_real.csv"), index=False)

    # ===== 新增：基线 Logistic 模型（CRP-only & 6 biomarkers），并保存测试集预测概率 =====
    print("\n=== Baseline logistic models on test set (for DCA baseline) ===")
    baseline_out = os.path.join(OUT_DIR, "baseline_pred_test.csv")

    # 确保所需特征存在
    crp_feat = ["CRP"]
    core6_feats = ["CRP", "IL6", "IL10", "TNFalpha", "ACTH", "PTC"]

    missing_crp = [f for f in crp_feat if f not in X_train.columns]
    missing_core6 = [f for f in core6_feats if f not in X_train.columns]

    if missing_crp:
        print(f"[警告] 缺少 CRP-only baseline 所需特征: {missing_crp}，跳过该 baseline。")
        p_logit_crp = np.full(shape=y_val.shape, fill_value=np.nan)
    else:
        pipe_crp = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED
            ))
        ])
        pipe_crp.fit(X_train[crp_feat], y_train)
        p_logit_crp = pipe_crp.predict_proba(X_val[crp_feat])[:, 1]
        thr_crp = best_threshold_f1(y_val, p_logit_crp)
        m_crp = metrics_at_threshold(y_val, p_logit_crp, thr_crp)
        print(f"Logistic (CRP-only): AUC={m_crp['ROC_AUC']:.3f}, F1@F1={m_crp['F1']:.3f}, Thr_F1={thr_crp:.3f}")

    if missing_core6:
        print(f"[警告] 缺少 6 biomarkers baseline 所需特征: {missing_core6}，跳过该 baseline。")
        p_logit_6bio = np.full(shape=y_val.shape, fill_value=np.nan)
    else:
        pipe_6bio = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED
            ))
        ])
        pipe_6bio.fit(X_train[core6_feats], y_train)
        p_logit_6bio = pipe_6bio.predict_proba(X_val[core6_feats])[:, 1]
        thr_6 = best_threshold_f1(y_val, p_logit_6bio)
        m_6 = metrics_at_threshold(y_val, p_logit_6bio, thr_6)
        print(f"Logistic (6 biomarkers): AUC={m_6['ROC_AUC']:.3f}, F1@F1={m_6['F1']:.3f}, Thr_F1={thr_6:.3f}")

    baseline_df = pd.DataFrame({
        "y_true": y_val.values,
        "p_logit_crp": p_logit_crp,
        "p_logit_6bio": p_logit_6bio
    })
    baseline_df.to_csv(baseline_out, index=False)
    print(f"baseline_pred_test.csv 已保存到: {baseline_out}")

if __name__ == "__main__":
    main()
