#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shap_analysis_from_final_report.py  —  A+B 全部输出版
python d:\yiqy\sleepProjexts\发文章版_疼痛二分 类任务\最终版本\shap_analysis_from_final_report.py --tasks pain --subsets blood_only --output_root D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap --n_splits 3 --kernel_bg 200 --kernel_nsamples 200
"""

import os, sys, glob, argparse, warnings, json, math, itertools
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.cluster import KMeans  # 新增：用于 SHAP-based endotype 聚类

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 140})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'

SUMMARY_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\stacking_model_comparison_summary.csv"
ENSEMBLE_PKL = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\final_ensemble_convex_T.pkl"
DEFAULT_OUTPUT_ROOT = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap"

# 反序列化所需：与保存时一致的集成模型类
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

# ================== 与对比脚本一致的前处理组件 ==================

class MissingIndicatorTransformer:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.missing_columns: List[str] = []
    def fit(self, X, y=None):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        miss = Xdf.isna().mean()
        self.missing_columns = miss[miss > self.threshold].index.tolist()
        return self
    def transform(self, X):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for c in self.missing_columns:
            if c in Xdf.columns:
                Xdf[f"{c}_missing"] = Xdf[c].isna().astype(int)
        return Xdf

class Log1pTransformer:
    def __init__(self, skew_threshold=1.0):
        self.skew_threshold = skew_threshold
        self.transform_columns: List[str] = []
    def fit(self, X, y=None):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        num_cols = Xdf.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if Xdf[c].min() >= 0 and abs(Xdf[c].skew()) > self.skew_threshold:
                self.transform_columns.append(c)
        return self
    def transform(self, X):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for c in self.transform_columns:
            if c in Xdf.columns:
                Xdf[c] = np.log1p(Xdf[c])
        return Xdf

class CustomVASImputer:
    def __init__(self, vas_col='VAS_score', pain_col='Chronic_pain'):
        self.vas_col = vas_col
        self.pain_col = pain_col
        self.median_pain: float = 0.0
    def fit(self, X, y=None):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.vas_col in Xdf.columns and self.pain_col in Xdf.columns:
            pain_mask = Xdf[self.pain_col] == 1
            vals = Xdf.loc[pain_mask, self.vas_col].astype(float)
            self.median_pain = float(vals.median()) if vals.notna().any() else 0.0
        return self
    def transform(self, X):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        if self.vas_col in Xdf.columns:
            if self.pain_col in Xdf.columns:
                pain_mask = Xdf[self.pain_col] == 1
                Xdf.loc[pain_mask & Xdf[self.vas_col].isna(), self.vas_col] = self.median_pain
                Xdf.loc[(~pain_mask) & Xdf[self.vas_col].isna(), self.vas_col] = 0.0
            else:
                med = Xdf[self.vas_col].astype(float).median()
                Xdf[self.vas_col] = Xdf[self.vas_col].fillna(med)
        return Xdf

def infer_column_types_for_ct(X: pd.DataFrame,
                              small_cardinality_threshold: int = 10) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in X.columns:
        s = X[c]
        if s.dtype == 'O' or str(s.dtype).startswith('category'):
            cat_cols.append(c); continue
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            nunq = s.dropna().nunique()
            (cat_cols if nunq <= small_cardinality_threshold else num_cols).append(c)
        else:
            num_cols.append(c)
    return num_cols, cat_cols

def make_preprocessor(X: pd.DataFrame, use_standard_scaler: bool) -> Pipeline:
    prefix = Pipeline([
        ('vas', CustomVASImputer()),
        ('miss', MissingIndicatorTransformer(threshold=0.05)),
        ('log1p', Log1pTransformer(skew_threshold=1.0))
    ])
    Xp = prefix.fit_transform(X)
    num_cols, cat_cols = infer_column_types_for_ct(Xp)
    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()) if use_standard_scaler else ('passthrough', 'passthrough')
    ])
    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
    ct = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop')
    return Pipeline([('prefix', prefix), ('ct', ct)])

# ================ 模型配置（与对比脚本一致） =================

def build_model_configs() -> Dict[str, Dict]:
    return {
        'Logistic Regression': {
            'cls': LogisticRegression, 'type': 'linear', 'scale': True,
            'base': dict(random_state=42, max_iter=1000),
            'grid': {'C': np.logspace(-3,3,7), 'penalty':['l2'], 'solver':['lbfgs','liblinear'],
                     'class_weight':['balanced',{0:1,1:2},{0:1,1:3}]}
        },
        'SVM': {
            'cls': SVC, 'type': 'svm', 'scale': True,
            'base': dict(random_state=42, probability=True),
            'grid': {'C':[0.1,1,10], 'kernel':['linear','rbf'], 'gamma':['scale',0.001,0.01,0.1],
                     'class_weight':['balanced',{0:1,1:2}]}
        },
        'Random Forest': {
            'cls': RandomForestClassifier, 'type': 'tree', 'scale': False,
            'base': dict(random_state=42),
            'grid': {'bootstrap':[True,False], 'max_depth':[6,10,20,50,80,90,100,200,None],
                     'max_features':['log2','sqrt'], 'min_samples_leaf':[1,2,4],
                     'min_samples_split':[2,5,10], 'n_estimators':[50,100,200,400,800,1200,1600,2000],
                     'class_weight':['balanced']}
        },
        'XGBoost': {
            'cls': XGBClassifier, 'type': 'tree', 'scale': False,
            'base': dict(random_state=42, eval_metric='logloss', verbosity=0),
            'grid': {'min_child_weight':[1,5,10], 'gamma':[0,0.1,0.5,1,1.5,2,5],
                     'subsample':[0.6,0.8,1.0], 'colsample_bytree':[0.6,0.8,1.0],
                     'max_depth':[3,4,5,6,10], 'n_estimators':[100,200,500,1000],
                     'learning_rate':[0.01,0.05,0.1,0.2], 'reg_lambda':[1e-5,1e-2,0.1,1,10,100],
                     'reg_alpha':[1e-5,1e-2,0.1,1,10,100]}
        },
        'Gradient Boosting': {
            'cls': GradientBoostingClassifier, 'type': 'tree', 'scale': False,
            'base': dict(random_state=42),
            'grid': {'max_depth':[3,6,9], 'learning_rate':[0.01,0.1,0.2], 'n_estimators':[100,200,500,1000],
                     'subsample':[0.6,0.8,1.0], 'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4]}
        },
        'LightGBM': {
            'cls': LGBMClassifier, 'type': 'tree', 'scale': False,
            'base': dict(random_state=42, verbosity=-1),
            'grid': {'max_depth':[3,6,9,15], 'learning_rate':[0.01,0.05,0.1,0.2],
                     'n_estimators':[100,200,500,1000], 'subsample':[0.6,0.8,1.0],
                     'colsample_bytree':[0.6,0.8,1.0], 'num_leaves':[31,50,100,200],
                     'min_child_samples':[20,50,100], 'reg_alpha':[0,0.1,0.5,1], 'reg_lambda':[0,0.1,0.5,1]}
        },
        'CatBoost': {
            'cls': CatBoostClassifier, 'type': 'tree', 'scale': False,
            'base': dict(random_state=42, verbose=False, early_stopping_rounds=50),
            'grid': {'depth':[3,6,9,12], 'learning_rate':[0.01,0.05,0.1,0.2], 'iterations':[100,200,300,500],
                     'l2_leaf_reg':[1,3,5,10], 'subsample':[0.6,0.8,1.0], 'colsample_bylevel':[0.6,0.8,1.0],
                     'border_count':[32,64,128]}
        },
        'Neural Network': {
            'cls': MLPClassifier, 'type': 'neural', 'scale': True,
            'base': dict(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                         alpha=0.01, learning_rate='adaptive', max_iter=500,
                         random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
            'grid': {'alpha':[0.0001,0.001,0.01],}
        }
    }

# ============== 工具：数据、子集、血液比值、分组 ==============

TASK_CONFIGS = {
    'anxiety': {'path': 'dataset/sleep_15_20%缺失值.csv', 'target': 'Anxiety_14',
                'blood_extra': 'Depression_18'},
    'depression': {'path': 'dataset/sleep_15_20%缺失值.csv', 'target': 'Depression_18',
                   'blood_extra': 'Anxiety_14'},
    'pain': {'path': r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv",
             'target': 'Chronic_pain',
             'blood_extra': ''}
}

BASE_BLOOD = ['IL6','IL10','TNFalpha','CRP','ACTH','PTC']
RATIO_PAIRS = [
    ('IL6','IL10'),('TNFalpha','IL10'),('CRP','IL10'),
    ('PTC','ACTH'),('PTC','IL6'),('PTC','CRP'),
    ('IL6','TNFalpha'),('CRP','IL6'),('ACTH','IL6')
]

def safe_div_col(df, a, b):
    if a in df.columns and b in df.columns:
        s = df[a].astype(float) / df[b].replace(0, np.nan).astype(float)
        return s.replace([np.inf,-np.inf], np.nan)
    return None

def ensure_blood_columns(task: str, df: pd.DataFrame) -> List[str]:
    cols = BASE_BLOOD.copy()
    for a,b in RATIO_PAIRS:
        col = f"{a}/{b}"
        if col not in df.columns:
            s = safe_div_col(df, a, b)
            if s is not None:
                df[col] = s
        cols.append(col)
    extra = TASK_CONFIGS[task]['blood_extra']
    cols = [c for c in cols if c != extra]
    return [c for c in cols if c in df.columns]

def feature_groups(task: str, df: pd.DataFrame) -> Dict[str, List[str]]:
    demo = ['Sex','Age','BMI','Smoking_status','Drinking_status','Tea_drinking_status',
            'Coffee_drinking_status','Education','Occupation','Marriage_status','Activity_level',
            'ACEs_total_score_11']
    clinical = ['Health_status_3groups','Insomnia_duration','ISI_total_score','PSQI_total_score',
                'Epworth_total_score','MEQ5_severity','Chalder_14_total_score',
                'Cognition_screening','Chronic_pain','VAS_score']
    blood = ensure_blood_columns(task, df)
    def keep_exist(a): return [c for c in a if c in df.columns]
    return {
        'blood_only': keep_exist(blood),
        'blood_six_only': keep_exist(BASE_BLOOD),
        'clinical_scales_only': keep_exist(clinical),
        'demographics_only': keep_exist(demo),
        'clinical_plus_blood': keep_exist(list(set(clinical + blood))),
        'clinical_plus_demo': keep_exist(list(set(clinical + demo))),
        'demo_plus_blood': keep_exist(list(set(demo + blood))),
        'clinical_plus_demo_plus_blood': keep_exist(list(set(demo + clinical + blood)))
    }

# ============== 结果目录&最佳模型选择 ==============

def latest_run_id(results_root: str) -> Optional[str]:
    if not os.path.isdir(results_root): return None
    cands = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root,d))]
    return sorted(cands)[-1] if cands else None

def pick_best_model(summary_csv: str) -> str:
    df = pd.read_csv(summary_csv)
    auc_col = 'roc_auc_mean' if 'roc_auc_mean' in df.columns else ('ROC_AUC_mean' if 'ROC_AUC_mean' in df.columns else None)
    pr_col  = 'pr_auc_mean'  if 'pr_auc_mean'  in df.columns else ('PR_AUC_mean'  if 'PR_AUC_mean'  in df.columns else None)
    if auc_col is None:
        raise RuntimeError(f"AUC列不存在: {summary_csv}")
    df = df.sort_values([auc_col, pr_col] if pr_col else [auc_col], ascending=False)
    return str(df.iloc[0]['Model'])

def top4_models(summary_csv: str) -> List[str]:
    df = pd.read_csv(summary_csv)
    auc_col = 'roc_auc_mean' if 'roc_auc_mean' in df.columns else 'ROC_AUC_mean'
    df = df.sort_values(auc_col, ascending=False)
    allowed = set(build_model_configs().keys())
    base_sorted = [m for m in df['Model'].values if m in allowed]
    return base_sorted[:4]

def select_base_models(summary_csv: str, prefer: List[str] = None, k: int = 2) -> List[str]:
    if prefer is None:
        prefer = ['SVM', 'CatBoost']
    df = pd.read_csv(summary_csv)
    auc_col = 'roc_auc_mean' if 'roc_auc_mean' in df.columns else ('ROC_AUC_mean' if 'ROC_AUC_mean' in df.columns else None)
    if auc_col is None:
        raise RuntimeError(f"AUC列不存在: {summary_csv}")
    allowed = set(build_model_configs().keys())
    models = [m for m in df['Model'].values if m in allowed]
    bases = [m for m in prefer if m in models]
    if len(bases) >= k:
        return bases[:k]
    df_base = df[df['Model'].isin(allowed)].copy()
    df_base = df_base.sort_values(auc_col, ascending=False)
    for m in df_base['Model'].values:
        if m not in bases:
            bases.append(m)
        if len(bases) >= k:
            break
    return bases[:k]

def auc_weights(summary_csv: str, names: List[str]) -> Dict[str,float]:
    df = pd.read_csv(summary_csv)
    auc_col = 'roc_auc_mean' if 'roc_auc_mean' in df.columns else 'ROC_AUC_mean'
    sub = df[df['Model'].isin(names)][['Model', auc_col]]
    tot = sub[auc_col].sum()
    return {r['Model']: (r[auc_col]/tot if tot>0 else 1.0/max(len(names),1)) for _,r in sub.iterrows()}

# ============== 训练单折模型（仅最佳模型用） ==============

def _mode_value(values: List[Any]) -> Any:
    if not values:
        return None
    try:
        from collections import Counter
        cnt = Counter(values)
        return cnt.most_common(1)[0][0]
    except Exception:
        return values[0]

def parse_detailed_best_params(run_id: str, task: str) -> Dict[str, Dict[str, Any]]:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', str(run_id))
    fpath = os.path.join(base_dir, task, 'detailed_model_results.txt')
    overrides: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(fpath):
        return overrides
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            txt = f.read()
        sections = {}
        for model in ['SVM','CatBoost']:
            i = txt.find(f"模型: {model}")
            if i < 0:
                continue
            j = txt.find('模型: ', i+1)
            seg = txt[i:j] if j > i else txt[i:]
            k = seg.find('各折最佳参数:')
            if k < 0:
                continue
            body = seg[k:]
            import re
            dicts = re.findall(r"\{[^\}]*\}", body)
            vals_by_key: Dict[str, List[Any]] = {}
            for d in dicts:
                items = re.findall(r"'([^']+)'\s*:\s*([^,}]+)", d)
                for key, val in items:
                    val = val.strip()
                    if val.startswith('np.float64(') and val.endswith(')'):
                        try:
                            val = float(val[len('np.float64('):-1])
                        except Exception:
                            pass
                    if val.startswith("'") and val.endswith("'"):
                        val = val[1:-1]
                    try:
                        if isinstance(val, str):
                            if '.' in val or 'e' in val.lower():
                                val_cast = float(val)
                            else:
                                val_cast = int(val)
                            val = val_cast
                    except Exception:
                        pass
                    vals_by_key.setdefault(key, []).append(val)
            if model == 'SVM':
                keys = ['kernel','C','gamma','class_weight']
            else:
                keys = ['iterations','depth','learning_rate','l2_leaf_reg','subsample','colsample_bylevel','border_count']
            agg: Dict[str, Any] = {}
            for k in keys:
                agg_val = _mode_value(vals_by_key.get(k, []))
                if agg_val is not None:
                    agg[k] = agg_val
            sections[model] = agg
        return sections
    except Exception:
        return overrides

def fit_best_model_one_fold(model_name: str, Xtr, ytr, use_scaler: bool, configs: Dict[str,Dict], overrides: Optional[Dict[str, Any]] = None):
    cfg = configs[model_name]
    pre = make_preprocessor(Xtr, use_scaler)
    Xtr_p = pre.fit_transform(Xtr)
    base = cfg['base'].copy()
    cls  = cfg['cls']
    grid = cfg['grid']
    if overrides and len(overrides) > 0:
        try:
            best = cls(**{**base, **overrides})
            best.fit(Xtr_p, ytr)
            return pre, best
        except Exception:
            pass
    inner = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=42)
    rs = RandomizedSearchCV(cls(**base), param_distributions=grid, n_iter=min(12, max(3, len(grid))),
                            cv=inner, scoring='roc_auc', n_jobs=-1, random_state=42, refit=True, verbose=0)
    rs.fit(Xtr_p, ytr)
    best = rs.best_estimator_
    return pre, best

def predict_proba_model(pre, est, X):
    Xp = pre.transform(X)
    if hasattr(est, "predict_proba"):
        return est.predict_proba(Xp)[:,1]
    else:
        dec = est.decision_function(Xp)
        mn, mx = dec.min(), dec.max()
        return (dec - mn) / (mx - mn + 1e-9)

# ============== SHAP：按模型类型选择解释器 =================

def kernel_shap_for_function(f, X_bg: pd.DataFrame, X_eval: pd.DataFrame,
                             nsamples: int = 1000, random_state: int = 42):
    np.random.seed(random_state)
    bg = X_bg.copy()
    explainer = shap.KernelExplainer(f, bg, link="identity")
    phi = explainer.shap_values(X_eval, nsamples=nsamples, l1_reg="num_features(10)")
    if isinstance(phi, list):
        phi = phi[1]
    exp = shap.Explanation(values=phi,
                           base_values=np.full(X_eval.shape[0], explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1]),
                           data=X_eval.values,
                           feature_names=list(X_eval.columns))
    return exp

def tree_or_linear_shap(pre, est, X_eval: pd.DataFrame):
    Xp = pre.transform(X_eval)
    X_pref = pre.named_steps['prefix'].transform(X_eval)
    ct: ColumnTransformer = pre.named_steps['ct']
    try:
        tf_names = ct.get_feature_names_out()
    except Exception:
        tf_names = [f'f{i}' for i in range(Xp.shape[1])]

    if any(k in est.__class__.__name__.lower() for k in ['forest','boost','xgb','lgbm','catboost','gradientboost']):
        explainer = shap.TreeExplainer(est, feature_perturbation="interventional", model_output="raw")
    elif isinstance(est, LogisticRegression):
        explainer = shap.LinearExplainer(est, Xp, feature_perturbation="interventional")
    else:
        def f_raw(Z):
            return predict_proba_model(pre, est, pd.DataFrame(Z, columns=list(X_eval.columns)))
        return kernel_shap_for_function(f_raw, X_eval.sample(min(200, len(X_eval)), random_state=42), X_eval)

    sv = explainer.shap_values(Xp)
    if isinstance(sv, list):
        sv = sv[1] if len(sv)==2 else sv[0]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1] if len(np.atleast_1d(base))>1 else base[0]

    try:
        num_cols = ct.transformers_[0][2]
        cat_cols = ct.transformers_[1][2]
    except Exception:
        num_cols, cat_cols = [], []

    col_map: List[str] = []
    for c in num_cols:
        col_map.append(c)
    for c in cat_cols:
        ohe = ct.named_transformers_['cat'].named_steps['ohe']
        cats = ohe.categories_[cat_cols.index(c)] if hasattr(ohe, 'categories_') else []
        for _ in (cats if len(cats)>0 else [None]):
            col_map.append(c)

    df_sv = pd.DataFrame(sv, columns=tf_names)
    df_map = pd.Series(col_map, index=tf_names)

    agg_vals = {}
    for orig in df_map.unique():
        cols = df_map[df_map==orig].index.tolist()
        agg_vals[orig] = df_sv[cols].sum(axis=1).values

    agg_matrix = np.column_stack([agg_vals[k] for k in agg_vals.keys()])
    feature_names = list(agg_vals.keys())

    exp = shap.Explanation(values=agg_matrix,
                           base_values=np.full(X_eval.shape[0], base),
                           data=X_pref[feature_names].values,
                           feature_names=feature_names)
    return exp

# ============== 评估/抽样辅助 ==============

def best_f1_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, p)
    best_t, best_f = 0.5, -1
    for t in thr:
        pred = (p >= t).astype(int)
        f = f1_score(y_true, pred)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)

def pick_samples_for_waterfall(y_true: np.ndarray, p: np.ndarray, thr: float, k_border: int = 1):
    idx_all = np.arange(len(y_true))
    tp_idx = idx_all[(y_true==1)]
    tp = tp_idx[np.argmax(p[tp_idx])] if len(tp_idx)>0 else None
    tn_idx = idx_all[(y_true==0)]
    tn = tn_idx[np.argmin(p[tn_idx])] if len(tn_idx)>0 else None
    border = idx_all[np.argmin(np.abs(p - thr))] if len(idx_all)>0 else None
    picks = []
    if tp is not None: picks.append(('tp', int(tp)))
    if tn is not None: picks.append(('tn', int(tn)))
    if border is not None: picks.append(('borderline', int(border)))
    return picks
def pick_pain_mixed(exp_all: shap.Explanation, y_true: np.ndarray, p: np.ndarray, thr: float) -> Optional[int]:
    idx_all = np.arange(len(y_true))
    cand = idx_all[(y_true==1)]
    if cand.size == 0:
        return None
    vals = exp_all.values
    has_pos = (vals[cand] > 0).any(axis=1)
    has_neg = (vals[cand] < 0).any(axis=1)
    mix_mask = has_pos & has_neg
    cand_mix = cand[mix_mask]
    if cand_mix.size == 0:
        return int(cand[np.argmax(p[cand])])
    diffs = np.abs(p[cand_mix] - thr)
    return int(cand_mix[np.argmin(diffs)])

# ============== 作图 ==============

def plot_topk_bar(exp: shap.Explanation, out_png: str, topk: int = 20):
    vals = np.abs(exp.values).mean(axis=0)
    order = np.argsort(vals)[::-1][:topk]
    names = np.array(exp.feature_names)[order]
    scores = vals[order]
    pref_top = {"crp", "il6"}
    pref_bottom = {"CRP"}
    items = [(n, s) for n, s in zip(names, scores)]
    top_items = [(n, s) for n, s in items if str(n).lower() in pref_top]
    bottom_items = [(n, s) for n, s in items if str(n) in pref_bottom]
    middle_items = [(n, s) for n, s in items if (str(n).lower() not in pref_top and str(n) not in pref_bottom)]
    ordered = top_items + middle_items + bottom_items
    disp_names = [str(n).replace("TNFalpha", "TNFα") for n, _ in ordered][::-1]
    disp_scores = np.array([s for _, s in ordered][::-1])
    plt.figure()
    bars = plt.barh(disp_names, disp_scores, edgecolor='black', linewidth=0.75)
    plt.xlabel("mean(|SHAP|) on probability")
    ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    for b, v in zip(bars, disp_scores):
        plt.text(v, b.get_y() + b.get_height()/2.0, f"{v:.6f}", va='center', ha='left', fontsize=10)
    ax.invert_xaxis()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

def plot_beeswarm(exp: shap.Explanation, out_png: str, max_samples=1000, random_state=42):
    idx = np.arange(exp.values.shape[0])
    if len(idx) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=max_samples, replace=False)
    shap.plots.beeswarm(exp[idx], show=False, max_display=20)
    ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close()

def plot_group_bar(exp: shap.Explanation, task: str, out_png: str):
    demo = {'Sex','Age','BMI','Smoking_status','Drinking_status','Tea_drinking_status','Coffee_drinking_status',
            'Education','Occupation','Marriage_status','Activity_level','ACEs_total_score_11'}
    clinical = {'Health_status_3groups','Insomnia_duration','ISI_total_score','PSQI_total_score','Epworth_total_score',
                'MEQ5_severity','Chalder_14_total_score','Cognition_screening','Chronic_pain','VAS_score'}
    blood = set(BASE_BLOOD + [f"{a}/{b}" for a,b in RATIO_PAIRS])
    extra = TASK_CONFIGS[task]['blood_extra']
    if extra in blood: blood.remove(extra)
    vals = np.abs(exp.values).mean(axis=0)
    name2val = dict(zip(exp.feature_names, vals))
    gsum = {'Demographics':0.0, 'Clinical':0.0, 'Blood':0.0}
    for n,v in name2val.items():
        if n in demo: gsum['Demographics'] += v
        elif n in clinical: gsum['Clinical'] += v
        elif n in blood: gsum['Blood'] += v
    plt.figure(figsize=(5,4))
    plt.bar(list(gsum.keys()), list(gsum.values()), edgecolor='black', linewidth=0.75)
    plt.ylabel("mean(|SHAP|)")
    ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

def plot_cumulative_importance(exp: shap.Explanation, out_png: str):
    vals = np.abs(exp.values).mean(axis=0)
    order = np.argsort(vals)[::-1]
    s = np.cumsum(vals[order])
    s /= s[-1] if s[-1] != 0 else 1.0
    xs = np.arange(1, len(order)+1)
    plt.figure(figsize=(6,4))
    plt.plot(xs, s)
    for tgt in [0.8, 0.9]:
        k = int(np.argmax(s >= tgt)) + 1
        plt.axhline(tgt, ls='--', alpha=0.6)
        plt.axvline(k, ls='--', alpha=0.6)
        plt.text(k, tgt, f" {tgt*100:.0f}%@{k}")
    plt.xlabel("#Features (sorted by mean|SHAP|)"); plt.ylabel("Cumulative share")
    ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

def plot_decision(exp: shap.Explanation, out_png: str, max_samples: int = 80, max_features: int = 10, random_state: int = 42):
    idx = np.arange(exp.values.shape[0])
    if len(idx) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=max_samples, replace=False)
    try:
        shap.plots.decision(exp[idx][:, :max_features], show=False)
    except Exception:
        try:
            shap.decision_plot(base_value=np.mean(exp.base_values[idx]),
                               shap_values=exp.values[idx][:, :max_features],
                               feature_names=exp.feature_names[:max_features], show=False)
        except Exception:
            return
    ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close()

def plot_dependences(exp: shap.Explanation, out_dir: str, topm: int = 8):
    vals = np.abs(exp.values).mean(axis=0)
    order = np.argsort(vals)[::-1][:topm]
    names = np.array(exp.feature_names)[order]
    try:
        pair_indices = {}
        for i, f in enumerate(names):
            j_list = shap.approximate_interactions(f, exp.values, exp.data, feature_names=list(exp.feature_names))
            if len(j_list) > 1:
                pair_indices[f] = j_list[1]
            else:
                pair_indices[f] = None
    except Exception:
        pair_indices = {f: None for f in names}

    for f in names:
        f2_idx = pair_indices.get(f, None)
        try:
            if f2_idx is not None and 0 <= f2_idx < len(exp.feature_names):
                f2 = exp.feature_names[f2_idx]
                shap.plots.scatter(exp[:, f], color=exp[:, f2], show=False)
                plt.title(f"{str(f).replace('TNFalpha','TNFα')} (color by {str(f2).replace('TNFalpha','TNFα')})")
                fn = os.path.join(out_dir, f"dependence_{f.replace('/', '_')}__color_{f2.replace('/', '_')}.png")
            else:
                shap.plots.scatter(exp[:, f], show=False)
                plt.title(str(f).replace('TNFalpha','TNFα'))
                fn = os.path.join(out_dir, f"dependence_{f.replace('/', '_')}.png")
            ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(fn, dpi=600, bbox_inches='tight')
            plt.close()
        except Exception:
            try:
                fi = list(exp.feature_names).index(f)
                shap.dependence_plot(ind=fi, shap_values=exp.values, features=pd.DataFrame(exp.data, columns=exp.feature_names),
                                     interaction_index=f2_idx if isinstance(f2_idx, int) else 'auto', show=False)
                fn = os.path.join(out_dir, f"dependence_{f.replace('/', '_')}_legacy.png")
                ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.savefig(fn, dpi=600, bbox_inches='tight')
                plt.close()
            except Exception:
                continue

    pd.DataFrame({
        'feature': list(names),
        'paired_with': [exp.feature_names[pair_indices[f]] if isinstance(pair_indices.get(f), int) else '' for f in names]
    }).to_csv(os.path.join(out_dir, 'interaction_pairs.csv'), index=False, encoding='utf-8')

def plot_shap_abs_hist(exp: shap.Explanation, out_png: str):
    vals = np.abs(exp.values).ravel()
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=50, alpha=0.8, edgecolor='black', linewidth=0.75)
    plt.xlabel("|SHAP| on probability"); plt.ylabel("Count")
    ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()

# ============== 主流程 ==============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', type=str, default='results')
    ap.add_argument('--run_id', type=str, default='stacking_competition')
    ap.add_argument('--tasks', nargs='+', default=['pain'])
    ap.add_argument('--subsets', nargs='+', default=['blood_only'])
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--kernel_bg', type=int, default=200, help='KernelSHAP 背景样本数')
    ap.add_argument('--kernel_nsamples', type=int, default=1000, help='KernelSHAP nsamples')
    ap.add_argument('--max_beeswarm_samples', type=int, default=800)
    ap.add_argument('--analysis_set', type=str, choices=['train','test'], default='test')
    ap.add_argument('--topk_bar', type=int, default=20)
    ap.add_argument('--dependence_topm', type=int, default=8)
    ap.add_argument('--decision_max_samples', type=int, default=80)
    ap.add_argument('--decision_max_features', type=int, default=10)
    ap.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT, help='SHAP 结果输出根目录')
    ap.add_argument('--final_report_md', type=str, default=r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\stacking_model_comparison_summary.csv")
    ap.add_argument('--use_final_report_best', action='store_true', default=False,
                    help='从 final_report.md 解析每个任务的最佳模型')
    args = ap.parse_args()

    run_id = 'stacking_competition'

    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    model_cfgs = build_model_configs()

    best_by_task: Dict[str, str] = {'pain': 'Ensemble_Logit_T'}
    def parse_best_from_md(md_path: str) -> Dict[str, str]:
        if not os.path.isfile(md_path):
            return best_by_task.copy()
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                txt = f.read()
            res = best_by_task.copy()
            def extract(section_header: str) -> Optional[str]:
                i = txt.find(section_header)
                if i < 0:
                    return None
                j = txt.find('###', i+1)
                seg = txt[i:j] if j > i else txt[i:]
                k = seg.find('最佳模型')
                if k < 0:
                    return None
                line = seg[k: k+200]
                for token in ['Ensemble_Rank_Plat','Ensemble_Logit_T','Logistic Regression','SVM','Random Forest',
                              'CatBoost','XGBoost','LightGBM','Neural Network']:
                    if token in line:
                        return token
                return None
            dep_best = extract('DEPRESSION 任务')
            anx_best = extract('ANXIETY 任务')
            if dep_best: res['depression'] = dep_best
            if anx_best: res['anxiety'] = anx_best
            return res
        except Exception:
            return best_by_task.copy()

    if args.use_final_report_best:
        best_by_task = parse_best_from_md(args.final_report_md)
    try:
        csv_best = pick_best_model(SUMMARY_CSV)
        if ('logit_convex+T' in csv_best) or ('Logit_T' in csv_best):
            best_by_task['pain'] = 'Ensemble_Logit_T'
        elif ('rank' in csv_best) and ('Platt' in csv_best):
            best_by_task['pain'] = 'Ensemble_Rank_Plat'
        else:
            best_by_task['pain'] = csv_best
    except Exception:
        pass
    print(f"[BestModel] pain = {best_by_task['pain']}")

    for task in args.tasks:
        csv_path = TASK_CONFIGS[task]['path']
        target = TASK_CONFIGS[task]['target']
        if not os.path.exists(csv_path):
            print(f"[warn] 数据不存在: {csv_path}，跳过 {task}")
            continue
        for enc in ['utf-8','gbk','gb2312','latin1']:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if target not in df.columns:
            print(f"[warn] 目标列缺失: {target}，跳过 {task}")
            continue

        groups = feature_groups(task, df)

        for subset in args.subsets:
            best = best_by_task.get(task, 'Logistic Regression')

            if task == 'pain':
                try:
                    best_csv = pick_best_model(SUMMARY_CSV)
                    if ('logit_convex+T' in best_csv) or ('Logit_T' in best_csv):
                        best = 'Ensemble_Logit_T'
                    elif ('rank' in best_csv) and ('Platt' in best_csv):
                        best = 'Ensemble_Rank_Plat'
                    else:
                        best = best_csv
                except Exception:
                    pass

            feats = groups.get(subset, [])
            if len(feats)==0:
                print(f"[skip] {task}-{subset} 无特征")
                continue

            print(f"\n=== {task} | {subset} | Best: {best} ===")
            X = df[feats].copy()
            y = df[target].astype(int).copy()

            out_dir = output_root
            os.makedirs(out_dir, exist_ok=True)

            overrides_map = {}
            summary_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', str(run_id), task, 'summary.csv')

            skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
            exps: List[shap.Explanation] = []
            oof_p, oof_y = [], []

            if best == 'ExpertVoting_Top4':
                names = top4_models(summary_csv)
                weights = auc_weights(summary_csv, names)
                print(f"Top4 = {names}  | weights = {weights}")

                for tr, va in skf.split(X, y):
                    Xtr, Xva = X.iloc[tr], X.iloc[va]
                    ytr, yva = y.iloc[tr], y.iloc[va]
                    pipes = []
                    for m in names:
                        cfg = model_cfgs[m]
                        pre, est = fit_best_model_one_fold(m, Xtr, ytr, cfg['scale'], model_cfgs, overrides_map.get(m))
                        pipes.append((m, cfg, pre, est))

                    def f_batch(Xraw):
                        Xraw = pd.DataFrame(Xraw, columns=X.columns)
                        s = np.zeros(len(Xraw))
                        for m,cfg,pre,est in pipes:
                            s += weights[m] * predict_proba_model(pre, est, Xraw)
                        return s

                    bg = Xtr.sample(min(args.kernel_bg, len(Xtr)), random_state=42)
                    exp_fold = kernel_shap_for_function(
                        f_batch, bg, Xva, nsamples=args.kernel_nsamples, random_state=42)
                    exps.append(exp_fold)

                    p_va = f_batch(Xva)
                    oof_p.append(p_va)
                    oof_y.append(yva.values)

            elif best in ('Ensemble_Rank_Plat', 'Ensemble_Logit_T') and task != 'pain':
                base_names = ['SVM', 'CatBoost']
                print(f"[Ensemble] {best} with bases = {base_names}")
                for tr, va in skf.split(X, y):
                    Xtr, Xva = X.iloc[tr], X.iloc[va]
                    ytr, yva = y.iloc[tr], y.iloc[va]
                    pipes = []
                    for m in base_names:
                        cfg = model_cfgs[m]
                        pre, est = fit_best_model_one_fold(m, Xtr, ytr, cfg['scale'], model_cfgs, overrides_map.get(m))
                        pipes.append((m, cfg, pre, est))

                    P_tr_cols = []
                    P_va_cols = []
                    for m,cfg,pre,est in pipes:
                        P_tr_cols.append(predict_proba_model(pre, est, Xtr))
                        P_va_cols.append(predict_proba_model(pre, est, Xva))
                    P_tr = np.column_stack(P_tr_cols)
                    P_va = np.column_stack(P_va_cols)

                    def f_batch_rank_platt(Xraw):
                        r_tr_a = pd.Series(P_tr[:, 0]).rank(method='average').values / float(len(P_tr))
                        r_tr_b = pd.Series(P_tr[:, 1]).rank(method='average').values / float(len(P_tr))
                        s_tr = (r_tr_a + r_tr_b) / 2.0
                        lr_cal = LogisticRegression(max_iter=1000, solver='lbfgs')
                        lr_cal.fit(s_tr.reshape(-1, 1), ytr)
                        Xraw = pd.DataFrame(Xraw, columns=X.columns)
                        cols = [predict_proba_model(pre, est, Xraw) for m,cfg,pre,est in pipes]
                        S = np.column_stack(cols)
                        r_a = pd.Series(S[:, 0]).rank(method='average').values / float(len(S))
                        r_b = pd.Series(S[:, 1]).rank(method='average').values / float(len(S))
                        s = (r_a + r_b) / 2.0
                        return lr_cal.predict_proba(s.reshape(-1,1))[:, 1]

                    def f_batch_logit_T(Xraw):
                        def sigmoid(z):
                            return 1.0 / (1.0 + np.exp(-z))
                        def combine(P, alpha_w, alpha_T):
                            w = np.exp(alpha_w) / np.sum(np.exp(alpha_w))
                            T = np.log1p(np.exp(alpha_T))
                            z = w[0] * np.log(P[:,0] / (1 - P[:,0] + 1e-12)) + \
                                w[1] * np.log(P[:,1] / (1 - P[:,1] + 1e-12))
                            return sigmoid(z / (T + 1e-12))
                        def log_loss(y_true, p):
                            p = np.clip(p, 1e-7, 1-1e-7)
                            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
                        def objective(params):
                            aw = params[:2]
                            aT = params[2]
                            p_hat = combine(P_tr, aw, aT)
                            return log_loss(ytr, p_hat)
                        init = np.array([0.0, 0.0, 0.0])
                        res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 200})
                        aw_opt = res.x[:2]
                        aT_opt = res.x[2]
                        Xraw = pd.DataFrame(Xraw, columns=X.columns)
                        cols = [predict_proba_model(pre, est, Xraw) for m,cfg,pre,est in pipes]
                        P = np.column_stack(cols)
                        return combine(P, aw_opt, aT_opt)

                    if best == 'Ensemble_Rank_Plat':
                        f_batch = f_batch_rank_platt
                    else:
                        f_batch = f_batch_logit_T

                    bg = Xtr.sample(min(args.kernel_bg, len(Xtr)), random_state=42)
                    exp_fold = kernel_shap_for_function(
                        f_batch, bg, Xva, nsamples=args.kernel_nsamples, random_state=42)
                    exps.append(exp_fold)

                    p_va = f_batch(Xva)
                    oof_p.append(p_va)
                    oof_y.append(yva.values)

            elif task == 'pain' and best in ('Ensemble_Rank_Plat','Ensemble_Logit_T'):
                try:
                    mdl_path = ENSEMBLE_PKL
                    ensemble = joblib.load(mdl_path)
                except Exception:
                    print(f"[warn] 未能加载集成模型: {mdl_path}")
                    continue
                train_csv = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\train.csv"
                test_csv  = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv"
                try:
                    train_df = pd.read_csv(train_csv)
                    test_df  = pd.read_csv(test_csv)
                except Exception as e:
                    print(f"[warn] 读取固定切分失败: {e}")
                    train_df = df.copy()
                    test_df  = df.copy()
                feats = groups.get('blood_only', [])
                if len(feats)==0:
                    print("[skip] pain-blood_only 无特征")
                    continue
                Xtr_six = train_df[feats].copy()
                Xva_six = test_df[feats].copy()
                y_test = test_df[target].astype(int).values
                try:
                    ct = ensemble.preprocess.named_steps['ct']
                    exp_cols = []
                    try:
                        exp_cols.extend(list(ct.transformers_[0][2]))
                    except Exception:
                        pass
                    try:
                        exp_cols.extend(list(ct.transformers_[1][2]))
                    except Exception:
                        pass
                    cols_union = set(train_df.columns).union(set(test_df.columns))
                    exp_cols = [c for c in exp_cols if c in cols_union]
                except Exception:
                    exp_cols = [c for c in test_df.columns if c != target]
                Xtr_full = train_df[exp_cols].copy()
                fill_map = {}
                for c in exp_cols:
                    if c in Xtr_six.columns:
                        fill_map[c] = None
                    else:
                        s = Xtr_full[c]
                        if pd.api.types.is_numeric_dtype(s):
                            fill_map[c] = float(s.astype(float).median())
                        else:
                            try:
                                fill_map[c] = s.mode(dropna=True).iloc[0]
                            except Exception:
                                fill_map[c] = s.dropna().iloc[0] if s.dropna().shape[0]>0 else ''
                if best == 'Ensemble_Rank_Plat':
                    def f_model(Xraw):
                        Xraw = pd.DataFrame(Xraw, columns=list(Xtr_six.columns))
                        Z = pd.DataFrame({c: (Xraw[c] if (c in Xraw.columns and fill_map[c] is None) else pd.Series([fill_map[c]]*len(Xraw))) for c in exp_cols})
                        try:
                            p_svm = ensemble.svm_model.predict_proba(Z)[:, 1]
                            p_cat = ensemble.cat_model.predict_proba(Z)[:, 1]
                            n = len(Z)
                            r1 = (np.argsort(np.argsort(p_svm)) + 1) / float(n)
                            r2 = (np.argsort(np.argsort(p_cat)) + 1) / float(n)
                            s = ((r1 + r2) / 2.0).reshape(-1, 1)
                            return ensemble.platt_lr.predict_proba(s)[:, 1]
                        except Exception:
                            p = ensemble.predict_proba(Z)
                            if isinstance(p, np.ndarray) and p.ndim==2 and p.shape[1]==2:
                                return p[:,1]
                            return np.array(p).ravel()
                else:
                    p_svm_tr = ensemble.svm_model.predict_proba(Xtr_full)[:, 1]
                    p_cat_tr = ensemble.cat_model.predict_proba(Xtr_full)[:, 1]
                    ytr = train_df[target].astype(int).values
                    def expit(z):
                        return 1.0 / (1.0 + np.exp(-z))
                    def safe_logit(p):
                        p = np.clip(p, 1e-7, 1-1e-7)
                        return np.log(p / (1 - p))
                    def log_loss(y_true, p):
                        p = np.clip(p, 1e-7, 1-1e-7)
                        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
                    def objective(params):
                        aw = params[:2]
                        aT = params[2]
                        w = np.exp(aw) / np.sum(np.exp(aw))
                        T = np.log1p(np.exp(aT))
                        z = w[0] * safe_logit(p_svm_tr) + w[1] * safe_logit(p_cat_tr)
                        p_hat = expit(z / (T + 1e-12))
                        return log_loss(ytr, p_hat)
                    init = np.array([0.0, 0.0, 0.0])
                    try:
                        from scipy.optimize import minimize
                        res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 200})
                        aw_opt = res.x[:2]
                        aT_opt = res.x[2]
                    except Exception:
                        aw_opt = init[:2]
                        aT_opt = init[2]
                    def f_model(Xraw):
                        Xraw = pd.DataFrame(Xraw, columns=list(Xtr_six.columns))
                        Z = pd.DataFrame({c: (Xraw[c] if (c in Xraw.columns and fill_map[c] is None) else pd.Series([fill_map[c]]*len(Xraw))) for c in exp_cols})
                        p_svm = ensemble.svm_model.predict_proba(Z)[:, 1]
                        p_cat = ensemble.cat_model.predict_proba(Z)[:, 1]
                        w = np.exp(aw_opt) / np.sum(np.exp(aw_opt))
                        T = np.log1p(np.exp(aT_opt))
                        z = w[0] * safe_logit(p_svm) + w[1] * safe_logit(p_cat)
                        return expit(z / (T + 1e-12))
                bg = Xtr_six.sample(min(args.kernel_bg, len(Xtr_six)), random_state=42)
                X_eval = Xva_six if args.analysis_set == 'test' else Xtr_six
                exp_fold = kernel_shap_for_function(
                    f_model, bg, X_eval, nsamples=args.kernel_nsamples, random_state=42)
                exps.append(exp_fold)
                p_va = f_model(X_eval)
                oof_p.append(p_va)
                oof_y.append(y_test if args.analysis_set == 'test' else train_df[target].astype(int).values)
                X = X_eval.copy()
            else:
                use_scaler = model_cfgs[best]['scale']
                for tr, va in skf.split(X, y):
                    Xtr, Xva = X.iloc[tr], X.iloc[va]
                    ytr, yva = y.iloc[tr], y.iloc[va]
                    pre, est = fit_best_model_one_fold(best, Xtr, ytr, use_scaler, model_cfgs, overrides_map.get(best))

                    is_tree = model_cfgs[best]['type']=='tree'
                    is_linear = isinstance(est, LogisticRegression)
                    if is_tree or is_linear:
                        exp_fold = tree_or_linear_shap(pre, est, Xva)
                    else:
                        def f_batch(Xraw):
                            return predict_proba_model(pre, est, pd.DataFrame(Xraw, columns=X.columns))
                        bg = Xtr.sample(min(args.kernel_bg, len(Xtr)), random_state=42)
                        exp_fold = kernel_shap_for_function(
                            f_batch, bg, Xva, nsamples=args.kernel_nsamples, random_state=42)
                    exps.append(exp_fold)

                    p_va = predict_proba_model(pre, est, Xva)
                    oof_p.append(p_va)
                    oof_y.append(yva.values)

            # —— 聚合 OOF Explanation（按特征名对齐） —— #
            target_names = [n for n in X.columns]
            aligned_vals = []
            aligned_data = []
            base_vals = np.hstack([np.atleast_1d(e.base_values) for e in exps])
            for e in exps:
                name_to_idx = {n: idx for idx, n in enumerate(e.feature_names)}
                vals_row = []
                data_row = []
                for n in target_names:
                    if n in name_to_idx:
                        idx = name_to_idx[n]
                        vals_row.append(e.values[:, idx])
                        data_row.append(e.data[:, idx])
                    else:
                        vals_row.append(np.zeros(e.values.shape[0]))
                        data_row.append(np.zeros(e.data.shape[0]))
                aligned_vals.append(np.column_stack(vals_row))
                aligned_data.append(np.column_stack(data_row))
            all_vals = np.vstack(aligned_vals)
            all_data = np.vstack(aligned_data)
            feature_names = target_names
            exp_all = shap.Explanation(values=all_vals, base_values=base_vals,
                                       data=all_data, feature_names=feature_names)

            # ——— 全局重要性 CSV ——— #
            mean_abs = np.abs(exp_all.values).mean(axis=0)
            df_imp = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
            df_imp.sort_values('mean_abs_shap', ascending=False, inplace=True)
            df_imp.to_csv(os.path.join(out_dir, 'global_importance_raw.csv'), index=False, encoding='utf-8')
            total_mean_abs = float(df_imp['mean_abs_shap'].sum())
            if total_mean_abs > 0:
                df_imp['share_of_total'] = df_imp['mean_abs_shap'] / total_mean_abs
            else:
                df_imp['share_of_total'] = 0.0
            df_imp.to_csv(os.path.join(out_dir, 'global_importance_share.csv'), index=False, encoding='utf-8')

            # 组汇总（Demographics / Clinical / Blood）
            demo = {'Sex','Age','BMI','Smoking_status','Drinking_status','Tea_drinking_status','Coffee_drinking_status',
                    'Education','Occupation','Marriage_status','Activity_level','ACEs_total_score_11'}
            clinical = {'Health_status_3groups','Insomnia_duration','ISI_total_score','PSQI_total_score','Epworth_total_score',
                        'MEQ5_severity','Chalder_14_total_score','Cognition_screening','Chronic_pain','VAS_score'}
            blood = set(BASE_BLOOD + [f"{a}/{b}" for a,b in RATIO_PAIRS])
            extra = TASK_CONFIGS[task]['blood_extra']
            if extra in blood: blood.remove(extra)
            name2val = dict(zip(feature_names, mean_abs))
            gsum = {'Demographics':0.0, 'Clinical':0.0, 'Blood':0.0}
            for n,v in name2val.items():
                if n in demo: gsum['Demographics'] += v
                elif n in clinical: gsum['Clinical'] += v
                elif n in blood: gsum['Blood'] += v
            pd.DataFrame({'group': list(gsum.keys()), 'mean_abs_shap': list(gsum.values())}) \
              .to_csv(os.path.join(out_dir, 'global_importance_grouped.csv'), index=False, encoding='utf-8')

            # 新增：Pathway-level SHAP（炎症轴 vs HPA/血小板轴）
            infl_base = ['CRP','IL6','IL10','TNFalpha']
            hpa_base  = ['ACTH','PTC']

            def classify_pathway(feat: str) -> str:
                if feat in infl_base:
                    return 'Inflammatory_markers'
                if feat in hpa_base:
                    return 'HPA_platelet_markers'
                has_infl = any(tok in feat for tok in infl_base)
                has_hpa  = any(tok in feat for tok in hpa_base)
                if has_infl and not has_hpa:
                    return 'Inflammatory_ratios'
                if has_hpa:
                    return 'HPA_platelet_ratios'
                return 'Other'

            pathway_sum: Dict[str,float] = {}
            pathway_count: Dict[str,int] = {}
            total_sum = float(sum(name2val.values())) if len(name2val) > 0 else 1.0
            for n, v in name2val.items():
                grp = classify_pathway(n)
                pathway_sum[grp] = pathway_sum.get(grp, 0.0) + float(v)
                pathway_count[grp] = pathway_count.get(grp, 0) + 1

            rows_path = []
            for grp, val in pathway_sum.items():
                rows_path.append({
                    'group': grp,
                    'total_mean_abs_shap': val,
                    'share_of_total': float(val / total_sum),
                    'n_features': int(pathway_count.get(grp, 0))
                })
            df_path = pd.DataFrame(rows_path)
            df_path.sort_values('total_mean_abs_shap', ascending=False, inplace=True)
            df_path.to_csv(os.path.join(out_dir, 'pathway_shap_summary.csv'),
                           index=False, encoding='utf-8')

            plt.figure()
            colors = plt.cm.Blues(np.linspace(0.45, 0.85, 4))
            data_vals = list(df_path['total_mean_abs_shap'])
            color_list = [colors[i % 4] for i in range(len(data_vals))]
            plt.bar(df_path['group'], data_vals, edgecolor='black', linewidth=0.75, color=color_list)
            plt.ylabel("Total mean(|SHAP|)")
            plt.xticks(rotation=45)
            ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
            plt.tight_layout()
            path_png = os.path.join(out_dir, 'pathway_shap_bar.png')
            plt.savefig(path_png, dpi=600)
            plt.close()

            # 图：Top-k Bar / Beeswarm / Group Bar / Cumulative
            plot_topk_bar(exp_all, os.path.join(out_dir, f'top{args.topk_bar}_bar.png'), topk=args.topk_bar)
            plot_beeswarm(exp_all, os.path.join(out_dir, 'beeswarm.png'), max_samples=args.max_beeswarm_samples)
            plot_cumulative_importance(exp_all, os.path.join(out_dir, 'cumulative_importance.png'))

            # ——— Waterfall（TP/TN/边界） ——— #
            oof_p = np.concatenate(oof_p); oof_y = np.concatenate(oof_y)
            if oof_p.shape[0] != oof_y.shape[0]:
                thr = best_f1_threshold(oof_y[:min(len(oof_y), len(oof_p))], oof_p[:min(len(oof_y), len(oof_p))])
            else:
                thr = best_f1_threshold(oof_y, oof_p)
            picks = pick_samples_for_waterfall(oof_y, oof_p, thr)
            for tag, idx in picks:
                try:
                    shap.plots.waterfall(exp_all[idx], show=False)
                    title = f"{tag.upper()}  y={int(oof_y[idx])}  p={oof_p[idx]:.3f}  thr(F1)={thr:.3f}"
                    plt.title(title)
                    base_png = os.path.join(out_dir, f'waterfall_{tag}.png')
                    base, _ = os.path.splitext(base_png)
                    ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
                    plt.xticks(rotation=0)
                    plt.tight_layout(); plt.savefig(base_png, dpi=600, bbox_inches='tight');
                    plt.savefig(base + ".tif", dpi=600, bbox_inches='tight'); plt.close()
                except Exception:
                    with open(os.path.join(out_dir, f'waterfall_{tag}.txt'), 'w', encoding='utf-8') as fw:
                        fw.write(f"Sample index={idx}, y={int(oof_y[idx])}, p={oof_p[idx]:.4f}, thr={thr:.4f}\n")
            try:
                idx_mix = pick_pain_mixed(exp_all, oof_y, oof_p, thr)
                if idx_mix is not None:
                    shap.plots.waterfall(exp_all[idx_mix], show=False)
                    title = f"PAIN_MIXED  y={int(oof_y[idx_mix])}  p={oof_p[idx_mix]:.3f}  thr(F1)={thr:.3f}"
                    plt.title(title)
                    base_png = os.path.join(out_dir, 'waterfall_pain_mixed.png')
                    base, _ = os.path.splitext(base_png)
                    ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
                    plt.xticks(rotation=0)
                    plt.tight_layout(); plt.savefig(base_png, dpi=600, bbox_inches='tight');
                    plt.savefig(base + ".tif", dpi=600, bbox_inches='tight'); plt.close()
            except Exception:
                pass

            # ——— Dependence / Decision / 直方图 ——— #
            plot_dependences(exp_all, out_dir, topm=args.dependence_topm)
            plot_decision(exp_all, os.path.join(out_dir, 'decision_plot.png'),
                          max_samples=args.decision_max_samples, max_features=args.decision_max_features)
            plot_shap_abs_hist(exp_all, os.path.join(out_dir, 'shap_abs_hist.png'))

            # 新增：SHAP-based endotype（仅在 pain 任务上做聚类摘要）
            if task == 'pain':
                try:
                    top_n_endotype = min(6, exp_all.values.shape[1])
                    order_idx = np.argsort(mean_abs)[::-1][:top_n_endotype]
                    shap_sub = exp_all.values[:, order_idx]
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
                    clusters = kmeans.fit_predict(shap_sub)
                    df_endo = pd.DataFrame({
                        'cluster': clusters.astype(int),
                        'y_true': oof_y.astype(int)
                    })
                    stats_rows = []
                    for c in sorted(df_endo['cluster'].unique()):
                        sub = df_endo[df_endo['cluster'] == c]
                        stats_rows.append({
                            'cluster': int(c),
                            'n_samples': int(len(sub)),
                            'pain_rate': float(sub['y_true'].mean())
                        })
                    df_stats = pd.DataFrame(stats_rows)
                    df_stats.to_csv(os.path.join(out_dir, 'shap_endotypes_summary.csv'),
                                    index=False, encoding='utf-8')
                except Exception as e:
                    print(f"[warn] SHAP endotype clustering failed: {e}")

            meta = {
                'task': task, 'subset': subset, 'best_model': best,
                'n_samples_exp': int(exp_all.values.shape[0]),
                'n_features_exp': int(exp_all.values.shape[1]),
                'kernel_bg': args.kernel_bg, 'kernel_nsamples': args.kernel_nsamples,
                'topk_bar': args.topk_bar, 'dependence_topm': args.dependence_topm,
                'decision_max_samples': args.decision_max_samples,
                'decision_max_features': args.decision_max_features,
                'waterfall_threshold': float(thr),
                'note': 'OOF SHAP aggregated; probability scale; blood group excludes cross-task label in plots'
            }
            with open(os.path.join(out_dir, 'shap_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"[OK] {task}-{subset}-{best} → 输出到 {out_dir}")

if __name__ == '__main__':
    main()
