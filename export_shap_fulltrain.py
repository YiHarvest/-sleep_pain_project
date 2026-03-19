# export_shap_fulltrain.py
import os, numpy as np, pandas as pd, joblib
import shap

TRAIN_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\train.csv"
PKL_PATH  = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\final_ensemble_rank_platt.pkl"
OUT_DIR   = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap_fulltrain"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "Chronic_pain"

# 你用于建模的15个特征（一定要和训练模型一致）
FEATURES_15 = [
    "CRP","ACTH","PTC/CRP","PTC","PTC/IL6","PTC/ACTH","ACTH/IL6","IL6/IL10",
    "IL6","IL10","TNFalpha","TNFalpha/IL10","CRP/IL10","IL6/TNFalpha","CRP/IL6"
]

df = pd.read_csv(TRAIN_CSV)
X = df[FEATURES_15].copy()

model = joblib.load(PKL_PATH)

try:
    ct = model.preprocess.named_steps['ct']
    exp_cols = []
    try:
        exp_cols.extend(list(ct.transformers_[0][2]))
    except Exception:
        pass
    try:
        exp_cols.extend(list(ct.transformers_[1][2]))
    except Exception:
        pass
    exp_cols = [c for c in exp_cols if c in df.columns]
except Exception:
    exp_cols = [c for c in df.columns if c != TARGET]

fill_map = {}
for c in exp_cols:
    if c in FEATURES_15:
        fill_map[c] = None
    else:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            try:
                fill_map[c] = float(s.astype(float).median())
            except Exception:
                fill_map[c] = float(pd.to_numeric(s, errors="coerce").median())
        else:
            try:
                fill_map[c] = s.mode(dropna=True).iloc[0]
            except Exception:
                fill_map[c] = s.dropna().iloc[0] if s.dropna().shape[0] > 0 else ''

bg = shap.sample(X, 80, random_state=42)

def f_proba(x):
    X15 = pd.DataFrame(x, columns=FEATURES_15)
    Z = pd.DataFrame({
        c: (X15[c] if (c in X15.columns and fill_map[c] is None)
            else pd.Series([fill_map[c]] * len(X15)))
        for c in exp_cols
    })
    p = model.predict_proba(Z)
    return p[:, 1] if (isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 2) else np.asarray(p).ravel()

explainer = shap.KernelExplainer(f_proba, bg)
sv = explainer.shap_values(X, nsamples=200)

sv = np.array(sv)
pd.DataFrame(X, columns=FEATURES_15).to_csv(os.path.join(OUT_DIR, "shap_X.csv"), index=False)
pd.DataFrame(sv, columns=FEATURES_15).to_csv(os.path.join(OUT_DIR, "shap_values.csv"), index=False)

imp = np.mean(np.abs(sv), axis=0)
pd.DataFrame({"feature": FEATURES_15, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)\
  .to_csv(os.path.join(OUT_DIR, "global_importance_raw.csv"), index=False)

print("[OK] Exported:", OUT_DIR)
