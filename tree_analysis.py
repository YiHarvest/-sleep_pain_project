import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
import shap

TEST_CSV_DEFAULT = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv"
MODEL_PKL_DEFAULT = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\final_ensemble_convex_T.pkl"
OUT_DIR_DEFAULT = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\tree_analysis"

BLOOD_15_FEATURES = [
    "IL6", "IL10", "TNFalpha", "CRP", "ACTH", "PTC",
    "IL6/IL10", "TNFalpha/IL10", "CRP/IL10",
    "PTC/ACTH", "PTC/IL6", "PTC/CRP",
    "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6",
]

class LogitConvexTEnsemble:
    def __init__(self, preprocess, svm_model, cat_model, w_opt, T_opt):
        self.preprocess = preprocess
        self.svm_model = svm_model
        self.cat_model = cat_model
        self.w_opt = np.array(w_opt)
        self.T_opt = T_opt
    def _safe_logit(self, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1.0 - p))
    def _expit(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    def predict_proba(self, X):
        p_svm = self.svm_model.predict_proba(X)[:, 1]
        p_cat = self.cat_model.predict_proba(X)[:, 1]
        z = self.w_opt[0] * self._safe_logit(p_svm) + self.w_opt[1] * self._safe_logit(p_cat)
        p1 = self._expit(z / (self.T_opt + 1e-12))
        return np.column_stack([1.0 - p1, p1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", default=TEST_CSV_DEFAULT)
    ap.add_argument("--model-pkl", default=MODEL_PKL_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-leaf", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.04)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.test_csv)
    X_top = df[BLOOD_15_FEATURES].copy()
    rename_map = {
        "TNFalpha": "TNFα",
        "TNFalpha/IL10": "TNFα / IL10",
        "IL6/TNFalpha": "IL6 / TNFα",
        "IL6": "IL6",
        "IL10": "IL10",
    }
    X_display = X_top.rename(columns=rename_map)
    feature_names_display = list(X_display.columns)

    ensemble = joblib.load(args.model_pkl)
    X_for_model = df.drop(columns=["Chronic_pain"])
    proba = ensemble.predict_proba(X_for_model)[:, 1]
    y_teacher = (proba >= args.threshold).astype(int)

    clf = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_leaf, random_state=args.random_state)
    clf.fit(X_display, y_teacher)

    tree = clf.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    stack = [(0, 0)]
    nodes = []
    usage_overall = {}
    usage_by_depth = {}
    max_depth = 0
    while stack:
        nid, dep = stack.pop()
        max_depth = max(max_depth, dep)
        fid = int(feature[nid])
        thr = float(threshold[nid])
        nodes.append((nid, dep, fid, thr))
        if fid >= 0:
            usage_overall[fid] = usage_overall.get(fid, 0) + 1
            dmap = usage_by_depth.get(fid, {})
            dmap[dep] = dmap.get(dep, 0) + 1
            usage_by_depth[fid] = dmap
            stack.append((children_left[nid], dep + 1))
            stack.append((children_right[nid], dep + 1))

    leaf_ids = clf.apply(X_display.values)
    leaves = np.unique(leaf_ids)
    leaf_info = []
    for leaf in leaves:
        idx = np.where(leaf_ids == leaf)[0]
        cnt = int(len(idx))
        pos_rate = float(y_teacher[idx].mean()) if cnt > 0 else 0.0
        if cnt > 0:
            sidx = int(idx[0])
        else:
            sidx = 0
        node_indicator = clf.decision_path(X_display.iloc[[sidx]].values)
        node_index = node_indicator.indices
        rules = []
        for nid in node_index:
            fid = int(feature[nid])
            if fid >= 0:
                thr = float(threshold[nid])
                fname = X_display.columns[fid]
                val = float(X_display.iloc[sidx, fid])
                direction = "<=" if val <= thr else ">"
                rules.append((fname, thr, direction))
        leaf_info.append({"leaf": int(leaf), "count": cnt, "pos_rate": pos_rate, "rules": rules})

    feature_thresholds = {}
    for nid, dep, fid, thr in nodes:
        if fid >= 0:
            fname = X_display.columns[fid]
            lst = feature_thresholds.get(fname, [])
            lst.append(thr)
            feature_thresholds[fname] = lst

    pairs = [("PTC", "CRP"), ("ACTH", "IL6")]
    interaction_surfaces = {}
    med = X_display.median()
    for a, b in pairs:
        xa = X_display[a].values
        xb = X_display[b].values
        xa_min, xa_max = float(np.min(xa)), float(np.max(xa))
        xb_min, xb_max = float(np.min(xb)), float(np.max(xb))
        gx = np.linspace(xa_min, xa_max, 100)
        gy = np.linspace(xb_min, xb_max, 100)
        Z = np.zeros((len(gy), len(gx)))
        base_row = med.copy()
        for i, yv in enumerate(gy):
            for j, xv in enumerate(gx):
                base_row[a] = xv
                base_row[b] = yv
                p = clf.predict_proba(pd.DataFrame([base_row], columns=X_display.columns))[0, 1]
                Z[i, j] = p
        interaction_surfaces[f"{a}__{b}"] = {"gx": gx, "gy": gy, "Z": Z}

    explainer = shap.TreeExplainer(clf)
    X_np = X_display.values
    sv_full = explainer.shap_values(X_np)
    if isinstance(sv_full, list) and len(sv_full) >= 2:
        sv = sv_full[1]
    else:
        sv = sv_full
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)) and np.size(base) >= 2:
        base_val = float(np.atleast_1d(base)[1])
    else:
        base_val = float(np.atleast_1d(base)[0])
    try:
        inter_vals = explainer.shap_interaction_values(X_np)
    except Exception:
        inter_vals = None

    artifacts = {
        "clf": clf,
        "X_train": X_display,
        "y_train": y_teacher,
        "feature_names": feature_names_display,
        "feature_usage_overall": usage_overall,
        "feature_usage_by_depth": usage_by_depth,
        "max_depth": int(max_depth),
        "nodes": nodes,
        "leaf_info": leaf_info,
        "feature_thresholds": feature_thresholds,
        "interaction_surfaces": interaction_surfaces,
        "shap_values": sv,
        "shap_base": base_val,
        "shap_interactions": inter_vals,
        "feature_index_map": {name: i for i, name in enumerate(X_display.columns)},
        "meta": {
            "TREE_MAX_DEPTH": args.max_depth,
            "MIN_SAMPLES_LEAF": args.min_leaf,
            "RANDOM_STATE": args.random_state,
            "F1_THRESHOLD": args.threshold,
            "BLOOD_15_FEATURES": BLOOD_15_FEATURES,
        },
    }
    out_path = os.path.join(args.out_dir, "surrogate_tree_artifacts.joblib")
    joblib.dump(artifacts, out_path)
    print(out_path)

if __name__ == "__main__":
    main()
