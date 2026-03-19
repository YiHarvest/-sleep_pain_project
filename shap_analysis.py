#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shap_analysis.py — 训练与SHAP计算脚本
功能：训练模型、计算SHAP值、生成数据汇总CSV，并保存SHAP Explanation对象供绘图使用。
结果保存至：outputs/shap_analysis
"""

import os, sys, glob, argparse, warnings, json, math, itertools
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
import shap
import joblib
from scipy.optimize import minimize

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

warnings.filterwarnings("ignore")

SUMMARY_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\stacking_model_comparison_summary.csv"
ENSEMBLE_PKL = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\final_ensemble_convex_T.pkl"
DEFAULT_OUTPUT_ROOT = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap_analysis"

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

# ================== 前处理组件 ==================

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

# ================ 模型配置 =================

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

def auc_weights(summary_csv: str, names: List[str]) -> Dict[str,float]:
    df = pd.read_csv(summary_csv)
    auc_col = 'roc_auc_mean' if 'roc_auc_mean' in df.columns else 'ROC_AUC_mean'
    sub = df[df['Model'].isin(names)][['Model', auc_col]]
    tot = sub[auc_col].sum()
    return {r['Model']: (r[auc_col]/tot if tot>0 else 1.0/max(len(names),1)) for _,r in sub.iterrows()}

# ============== 训练单折模型 ==============

def _mode_value(values: List[Any]) -> Any:
    if not values: return None
    try:
        from collections import Counter
        cnt = Counter(values)
        return cnt.most_common(1)[0][0]
    except Exception:
        return values[0]

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

# ============== SHAP ==============

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
    ap.add_argument('--analysis_set', type=str, choices=['train','test'], default='test')
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
    
    if args.use_final_report_best:
        # 简化逻辑，直接使用 best_by_task
        pass
        
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
        
        df = None
        for enc in ['utf-8','gbk','gb2312','latin1']:
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None or target not in df.columns:
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

            try:
                pd.DataFrame(exp_all.values).to_csv("shap_values.csv", index=False, encoding='utf-8')
                pd.DataFrame(exp_all.data, columns=exp_all.feature_names).to_csv("shap_X.csv", index=False, encoding='utf-8')
                pd.DataFrame(exp_all.values).to_csv(os.path.join(out_dir, "shap_values.csv"), index=False, encoding='utf-8')
                pd.DataFrame(exp_all.data, columns=exp_all.feature_names).to_csv(os.path.join(out_dir, "shap_X.csv"), index=False, encoding='utf-8')
            except Exception:
                pass

            # ——— 全局重要性 CSV ——— #
            mean_abs = np.abs(exp_all.values).mean(axis=0)
            df_imp = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
            df_imp.sort_values('mean_abs_shap', ascending=False, inplace=True)
            df_imp.to_csv(os.path.join(out_dir, 'global_importance_raw.csv'), index=False, encoding='utf-8')

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

            # Pathway-level SHAP
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

            # 保存结果到 joblib，供绘图脚本使用
            save_data = {
                'exp_all': exp_all,
                'task': task,
                'subset': subset,
                'best_model': best,
                'oof_p': np.concatenate(oof_p) if len(oof_p) > 0 else np.array([]),
                'oof_y': np.concatenate(oof_y) if len(oof_y) > 0 else np.array([]),
                'feature_names': feature_names,
                'run_id': run_id,
                'args': args,
                'df_path': df_path  # 用于pathway bar plot
            }
            save_path = os.path.join(out_dir, f'shap_data_{task}_{subset}.joblib')
            joblib.dump(save_data, save_path)
            print(f"结果已保存至: {save_path}")

if __name__ == '__main__':
    main()
