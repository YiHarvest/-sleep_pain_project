# -*- coding: utf-8 -*-
"""
surrogate_tree_testset_convex_T.py

构建 15 个血液指标的 surrogate decision tree，在外部测试集上近似
最终集成模型 "Ensemble: logit_convex+T (GB+RF)" 的预测。
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'

# =====================================================================
# 1. Configuration
# =====================================================================

TEST_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv"
TARGET_COL = "Chronic_pain"
MODEL_PKL = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\final_ensemble_convex_T.pkl"
GLOBAL_IMPORTANCE_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap\global_importance_raw.csv"

OUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\surrogate_tree_testset"
os.makedirs(OUT_DIR, exist_ok=True)

BLOOD_15_FEATURES = [
    "IL6", "IL10", "TNFalpha", "CRP", "ACTH", "PTC",
    "IL6/IL10", "TNFalpha/IL10", "CRP/IL10",
    "PTC/ACTH", "PTC/IL6", "PTC/CRP",
    "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6",
]

TREE_MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10
F1_THRESHOLD = 0.30
RANDOM_STATE = 42

# =====================================================================
# 2. Read test data
# =====================================================================

print(f"[INFO] Reading test set: {TEST_CSV}")
test_df = pd.read_csv(TEST_CSV)

if TARGET_COL not in test_df.columns:
    raise ValueError(f"Target column {TARGET_COL} not found in test CSV.")

y_test_true = test_df[TARGET_COL].astype(int).values

print("\n[INFO] Using predefined 15 blood biomarkers as tree features.")
missing = [f for f in BLOOD_15_FEATURES if f not in test_df.columns]
if missing:
    raise ValueError(f"The following blood features are not found in test data: {missing}")

top_features = BLOOD_15_FEATURES
print("\n[INFO] 15 features used in the surrogate tree:")
for i, f in enumerate(top_features, 1):
    print(f"  {i}. {f}")

X_test_top = test_df[top_features].copy()

# =====================================================================
# 3. Load ensemble model & compute test-set probabilities
# =====================================================================

print(f"\n[INFO] Loading final ensemble model: {MODEL_PKL}", flush=True)

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

_t0 = time.time()
ensemble = joblib.load(MODEL_PKL)
print(f"[OK] Ensemble loaded in {time.time()-_t0:.2f}s", flush=True)

X_test_for_model = test_df.drop(columns=[TARGET_COL])

print("[INFO] Computing ensemble predicted probabilities on the test set...", flush=True)
_t1 = time.time()
proba_test = ensemble.predict_proba(X_test_for_model)[:, 1]
print(f"[OK] Predicted probabilities computed in {time.time()-_t1:.2f}s", flush=True)

y_test_teacher = (proba_test >= F1_THRESHOLD).astype(int)
print(f"[INFO] Using F1 threshold {F1_THRESHOLD:.2f} to define teacher labels (0=low, 1=high).")
print(f"       Teacher high-risk rate on test set = {y_test_teacher.mean():.3f}")

# =====================================================================
# 4. Train surrogate decision tree (on test set only)
# =====================================================================

print("\n[INFO] Training surrogate decision tree on test set (15 blood biomarkers)...")
tree_clf = DecisionTreeClassifier(
    max_depth=TREE_MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE
)
tree_clf.fit(X_test_top, y_test_teacher)

y_tree_teacher_pred = tree_clf.predict(X_test_top)
y_tree_teacher_proba = tree_clf.predict_proba(X_test_top)[:, 1]

fidelity_acc = accuracy_score(y_test_teacher, y_tree_teacher_pred)
try:
    fidelity_auc = roc_auc_score(y_test_teacher, y_tree_teacher_proba)
except ValueError:
    fidelity_auc = np.nan
fidelity_f1 = f1_score(y_test_teacher, y_tree_teacher_pred)

print("\n[RESULT] Surrogate tree vs TEACHER (test set):")
print(f"  · Accuracy = {fidelity_acc:.3f}")
print(f"  · ROC AUC  = {fidelity_auc:.3f}")
print(f"  · F1       = {fidelity_f1:.3f}")

try:
    auc_real = roc_auc_score(y_test_true, y_tree_teacher_proba)
except ValueError:
    auc_real = np.nan
acc_real = accuracy_score(y_test_true, y_tree_teacher_pred)
f1_real = f1_score(y_test_true, y_tree_teacher_pred)

print("\n[RESULT] Surrogate tree vs TRUE LABEL (test set, descriptive only):")
print(f"  · Accuracy = {acc_real:.3f}")
print(f"  · ROC AUC  = {auc_real:.3f}")
print(f"  · F1       = {f1_real:.3f}")

metrics_txt = os.path.join(OUT_DIR, "surrogate_tree_testset_metrics.txt")
with open(metrics_txt, "w", encoding="utf-8") as f:
    f.write("Surrogate tree vs TEACHER (test set)\n")
    f.write(f"Accuracy = {fidelity_acc:.3f}\n")
    f.write(f"ROC_AUC  = {fidelity_auc:.3f}\n")
    f.write(f"F1       = {fidelity_f1:.3f}\n\n")
    f.write("Surrogate tree vs TRUE LABEL (test set)\n")
    f.write(f"Accuracy = {acc_real:.3f}\n")
    f.write(f"ROC_AUC  = {auc_real:.3f}\n")
    f.write(f"F1       = {f1_real:.3f}\n")
print(f"[INFO] Metrics written to: {metrics_txt}")

# 新增：简单性能对比表（树 vs 教师 vs 真实标签）
perf_csv = os.path.join(OUT_DIR, "surrogate_tree_perf_vs_ensemble.csv")
pd.DataFrame([
    {
        "comparison": "vs_teacher",
        "Accuracy": fidelity_acc,
        "ROC_AUC": fidelity_auc,
        "F1": fidelity_f1
    },
    {
        "comparison": "vs_true_label",
        "Accuracy": acc_real,
        "ROC_AUC": auc_real,
        "F1": f1_real
    }
]).to_csv(perf_csv, index=False, encoding="utf-8-sig")
print(f"[OK] Performance comparison saved to: {perf_csv}")

# =====================================================================
# 4b. Threshold-consistency checks (F1 / Youden / Recall80)
# =====================================================================

thr_grid = np.linspace(0.01, 0.99, 99)

def youden_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    rec = tp / (tp + fn + 1e-9)
    spe = tn / (tn + fp + 1e-9)
    return rec + spe - 1.0

def find_thr_f1(y_true, proba):
    best_thr = 0.5
    best_f1 = -1
    for thr in thr_grid:
        y_pred = (proba >= thr).astype(int)
        f1_val = f1_score(y_true, y_pred)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thr = thr
    return best_thr

def find_thr_youden(y_true, proba):
    best_thr = 0.5
    best_youden = -1
    for thr in thr_grid:
        y_pred = (proba >= thr).astype(int)
        ydn = youden_score(y_true, y_pred)
        if ydn > best_youden:
            best_youden = ydn
            best_thr = thr
    return best_thr

def find_thr_recall_target(y_true, proba, target=0.80):
    best_thr = 0.5
    best_diff = 1e9
    for thr in thr_grid:
        y_pred = (proba >= thr).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        rec = tp / (tp + fn + 1e-9)
        diff = abs(rec - target)
        if diff < best_diff:
            best_diff = diff
            best_thr = thr
    return best_thr

thr_f1 = find_thr_f1(y_test_teacher, y_tree_teacher_proba)
thr_youden = find_thr_youden(y_test_teacher, y_tree_teacher_proba)
thr_rec80 = find_thr_recall_target(y_test_teacher, y_tree_teacher_proba, target=0.80)

yt_f1 = (proba_test >= F1_THRESHOLD).astype(int)
yt_youden = (proba_test >= thr_youden).astype(int)
yt_rec80 = (proba_test >= thr_rec80).astype(int)
y_tree = y_tree_teacher_pred

agree_f1 = np.mean(yt_f1 == y_tree)
agree_youden = np.mean(yt_youden == y_tree)
agree_rec80 = np.mean(yt_rec80 == y_tree)
rho, pval = spearmanr(proba_test, y_tree_teacher_proba)

consistency_csv = os.path.join(OUT_DIR, "surrogate_tree_threshold_consistency.csv")
pd.DataFrame({
    "metric": ["F1", "Youden", "Recall80", "Spearman"],
    "value": [agree_f1, agree_youden, agree_rec80, rho]
}).to_csv(consistency_csv, index=False)
print(f"[INFO] Threshold-consistency summary saved to: {consistency_csv}")

# =====================================================================
# 5. Export full decision tree as Graphviz .dot + highlight trunk
# =====================================================================

dot_path = os.path.join(OUT_DIR, "surrogate_tree_15blood_test.dot")
print(f"\n[INFO] Exporting Graphviz DOT: {dot_path}")

export_graphviz(
    tree_clf,
    out_file=dot_path,
    feature_names=top_features,
    class_names=["Low risk", "High risk"],
    filled=True,
    rounded=True,
    special_characters=True,
)

print("[INFO] DOT exported. Now highlighting main trunk...")

def get_leaf_path_counts(clf: DecisionTreeClassifier, X: pd.DataFrame) -> Tuple[int, List[int]]:
    tree_ = clf.tree_
    leaf_ids = clf.apply(X)
    unique_leaf_ids, counts = np.unique(leaf_ids, return_counts=True)
    trunk_leaf_id = unique_leaf_ids[np.argmax(counts)]

    children_left = tree_.children_left
    children_right = tree_.children_right
    parent = {}
    for node_id in range(tree_.node_count):
        left = children_left[node_id]
        right = children_right[node_id]
        if left != -1:
            parent[left] = node_id
        if right != -1:
            parent[right] = node_id

    path = [trunk_leaf_id]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path = path[::-1]
    return trunk_leaf_id, path

trunk_leaf_id, trunk_path_nodes = get_leaf_path_counts(tree_clf, X_test_top)
print(f"[INFO] Trunk leaf (most samples) = node {trunk_leaf_id}")
print(f"[INFO] Trunk path nodes = {trunk_path_nodes}")

with open(dot_path, "r", encoding="utf-8") as f:
    dot_lines = f.readlines()

trunk_edges = set()
for i in range(len(trunk_path_nodes) - 1):
    parent_id = trunk_path_nodes[i]
    child_id = trunk_path_nodes[i+1]
    trunk_edges.add((parent_id, child_id))

new_lines = []
for line in dot_lines:
    stripped = line.strip()
    if "->" in stripped:
        try:
            parts = stripped.split("->")
            left = parts[0].strip()
            right = parts[1].split("[")[0].strip().strip(";")
            parent_id = int(left)
            child_id = int(right)
        except Exception:
            new_lines.append(line)
            continue

        if (parent_id, child_id) in trunk_edges:
            if "[" in line:
                line = line.replace(
                    "[",
                    '[color="darkred", penwidth=3, '
                )
            else:
                line = line.replace(
                    ";",
                    ' [penwidth=3, color="darkred"];'
                )

    new_lines.append(line)

with open(dot_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"[OK] Main trunk highlighted in DOT: {dot_path}")

try:
    import graphviz
    with open(dot_path, "r", encoding="utf-8") as f:
        src = graphviz.Source(f.read())
    pdf_base = os.path.join(OUT_DIR, "surrogate_tree_15blood_test")
    src.render(pdf_base, format="pdf", cleanup=True)
    print(f"[OK] PDF tree generated: {pdf_base}.pdf")
except Exception as e:
    print(f"[WARN] Graphviz rendering failed (you can open the .dot online): {e}")

# =====================================================================
# 6. Export leaf-node rule summary + risk bands
# =====================================================================

def extract_leaf_rules(
    clf: DecisionTreeClassifier,
    X: pd.DataFrame,
    feature_names: List[str],
    teacher_labels: np.ndarray
) -> pd.DataFrame:
    tree_ = clf.tree_
    children_left = tree_.children_left
    children_right = tree_.children_right
    thresholds = tree_.threshold
    features = tree_.feature

    leaf_node_ids = np.where(children_left == -1)[0]
    samples_leaf = clf.apply(X)

    rows = []
    for leaf_id in leaf_node_ids:
        idx = np.where(samples_leaf == leaf_id)[0]
        if len(idx) == 0:
            continue
        teacher_sub = teacher_labels[idx]
        high_rate = np.mean(teacher_sub)

        rule_clauses = []
        node_id = leaf_id
        parent = {}
        for nid in range(tree_.node_count):
            left = children_left[nid]
            right = children_right[nid]
            if left != -1:
                parent[left] = (nid, "left")
            if right != -1:
                parent[right] = (nid, "right")

        path = []
        while node_id in parent:
            p, direction = parent[node_id]
            path.append((p, direction))
            node_id = p
        path = path[::-1]

        for (p, direction) in path:
            feat_idx = features[p]
            thr = thresholds[p]
            if feat_idx < 0:
                continue
            feat_name = feature_names[feat_idx]
            if direction == "left":
                clause = f"{feat_name} <= {thr:.3f}"
            else:
                clause = f"{feat_name} > {thr:.3f}"
            rule_clauses.append(clause)

        rule_text = " & ".join(rule_clauses) if rule_clauses else "(root)"
        rows.append({
            "leaf_node_id": leaf_id,
            "n_samples": len(idx),
            "teacher_high_rate": high_rate,
            "rule": rule_text
        })

    df_leaf = pd.DataFrame(rows)

    # 新增：按 teacher_high_rate 分层 Low / Intermediate / High
    def assign_risk_band(p: float) -> str:
        if p >= 0.70:
            return "High"
        elif p >= 0.40:
            return "Intermediate"
        else:
            return "Low"

    df_leaf["risk_band"] = df_leaf["teacher_high_rate"].apply(assign_risk_band)
    df_leaf = df_leaf.sort_values("teacher_high_rate", ascending=False)
    return df_leaf

leaf_df = extract_leaf_rules(
    clf=tree_clf,
    X=X_test_top,
    feature_names=top_features,
    teacher_labels=y_test_teacher
)

summary_csv = os.path.join(OUT_DIR, "surrogate_tree_15blood_test_summary.csv")
try:
    leaf_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Leaf-node rule summary saved to: {summary_csv}")
except PermissionError:
    alt_csv = os.path.join(OUT_DIR, "surrogate_tree_15blood_test_summary_alt.csv")
    leaf_df.to_csv(alt_csv, index=False, encoding="utf-8-sig")
    print(f"[WARN] Permission denied writing summary; saved to: {alt_csv}")

# 新增：样本级风险分层表
samples_leaf = tree_clf.apply(X_test_top)
leaf_rate_map = leaf_df.set_index("leaf_node_id")["teacher_high_rate"].to_dict()
leaf_band_map = leaf_df.set_index("leaf_node_id")["risk_band"].to_dict()

sample_rows = []
for i in range(len(y_test_true)):
    lid = int(samples_leaf[i])
    rate = float(leaf_rate_map.get(lid, np.nan))
    band = leaf_band_map.get(lid, "NA")
    sample_rows.append({
        "y_true": int(y_test_true[i]),
        "p_ensemble": float(proba_test[i]),
        "p_tree": float(y_tree_teacher_proba[i]),
        "leaf_node_id": lid,
        "teacher_high_rate_leaf": rate,
        "risk_band": band
    })
df_samples = pd.DataFrame(sample_rows)
sample_csv = os.path.join(OUT_DIR, "surrogate_tree_testset_risk_scores.csv")
df_samples.to_csv(sample_csv, index=False, encoding="utf-8-sig")
print(f"[OK] Sample-level risk scores saved to: {sample_csv}")

# =====================================================================
# 7. Rule forest plot
# =====================================================================

top_n_rules = min(10, leaf_df.shape[0])
top = leaf_df.head(top_n_rules).copy()
top = top.sort_values("teacher_high_rate", ascending=True)

baseline = float(np.mean(y_test_true))

plt.figure(figsize=(7.5, max(4, 0.42 * top_n_rules)))
ys = np.arange(top_n_rules)

mus = top["teacher_high_rate"].values
ns = top["n_samples"].values.astype(float)
ci_l = []
ci_u = []
colors = []
for p, n in zip(mus, ns):
    se = np.sqrt((p * (1.0 - p)) / max(n, 1.0))
    lo = float(np.clip(p - 1.96 * se, 0.0, 1.0))
    hi = float(np.clip(p + 1.96 * se, 0.0, 1.0))
    ci_l.append(lo)
    ci_u.append(hi)
    colors.append("red" if p >= baseline else "blue")

for i, y in enumerate(ys):
    plt.hlines(y, ci_l[i], ci_u[i], color=colors[i], linewidth=2.2)
    plt.plot(mus[i], y, marker="o", color=colors[i], markersize=6)

plt.axvline(baseline, color="gray", linestyle="--", linewidth=1.2)

plt.yticks(
    ys,
    [f"Leaf {int(t)} (n={int(n)}, {rb})"
     for t, n, rb in zip(top["leaf_node_id"].values,
                         top["n_samples"].values,
                         top["risk_band"].values)],
    fontsize=10
)
plt.xlabel("Positive rate with 95% CI", fontsize=10)
plt.xlim(0.0, 1.0)
ax = plt.gca(); ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
plt.xticks(rotation=0, fontsize=10)
plt.tight_layout()

forest_png = os.path.join(OUT_DIR, "surrogate_tree_rule_forest.png")
base, _ = os.path.splitext(forest_png)
plt.savefig(forest_png, dpi=600)
plt.savefig(base + ".tif", dpi=600)
plt.close()

print(f"[OK] Rule forest plot saved: {forest_png}")
print("\n[DONE] Surrogate decision tree on test set (with risk bands) is ready.")
