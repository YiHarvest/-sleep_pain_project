import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score

# 配置路径与列名
PROJECT_ROOT = r"d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\机器学习-英文版1.49- IL-6 CRP"
DATA_PATH = os.path.join(PROJECT_ROOT, "datasetprocess_results", "cleaned_imputed_data_mice.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasetprocess_results", "univariate_metrics_mice.csv")

OUTCOMES = ["Chronic_pain", "Depression_18", "Anxiety_14"]
BIOMARKERS = ["IL6", "IL10", "TNFalpha", "CRP", "ACTH", "PTC"]


def compute_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """使用ROC曲线基于Youden指数选择最佳阈值。"""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return thresholds[best_idx]


def bootstrap_auc(y_true: np.ndarray, scores: np.ndarray, n_bootstrap: int = 2000, random_state: int = 42):
    """对ROC-AUC进行自助法(bootstrap)95%置信区间估计。返回 (low, high)。"""
    rng = np.random.default_rng(random_state)
    aucs = []
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        s_b = scores[idx]
        # 要求两个类别都出现
        if len(np.unique(y_b)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_b, s_b))
        except Exception:
            continue
    if not aucs:
        return np.nan, np.nan
    low = float(np.percentile(aucs, 2.5))
    high = float(np.percentile(aucs, 97.5))
    return low, high


def compute_binary_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
     """在给定阈值下计算敏感性、特异性、准确性、Precision/Recall/F1/PPV/NPV等。"""
     y_pred = (scores >= thr).astype(int)
     tp = np.sum((y_true == 1) & (y_pred == 1))
     tn = np.sum((y_true == 0) & (y_pred == 0))
     fp = np.sum((y_true == 0) & (y_pred == 1))
     fn = np.sum((y_true == 1) & (y_pred == 0))
 
     # 避免除零
     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
     specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
     accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
 
     precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # PPV
     recall = sensitivity
     npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
     f1 = (2 * precision * recall / (precision + recall)) if (not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0) else np.nan
 
     youden_index = sensitivity + specificity - 1 if not (np.isnan(sensitivity) or np.isnan(specificity)) else np.nan
 
     return {
         "sensitivity": sensitivity,
         "specificity": specificity,
         "accuracy": accuracy,
         "youden_index": youden_index,
         "precision": precision,
         "recall": recall,
         "f1": f1,
         "ppv": precision,
         "npv": npv,
     }


def bootstrap_metrics(y_true: np.ndarray, scores: np.ndarray, n_bootstrap: int = 2000, random_state: int = 42) -> dict:
     """对二分类相关指标与PR-AUC进行bootstrap 95%CI估计。返回字典：每项为(low, high)。"""
     rng = np.random.default_rng(random_state)
     n = len(y_true)
     if n < 2 or len(np.unique(y_true)) < 2:
         return {k: (np.nan, np.nan) for k in [
             "accuracy", "precision", "recall", "specificity", "f1", "pr_auc", "ppv", "npv", "youden_index"
         ]}
     acc = []; prec = []; rec = []; spec = []; f1s = []; pr = []; ppv = []; npvs = []; youden = []
     for _ in range(n_bootstrap):
         idx = rng.integers(0, n, n)
         y_b = y_true[idx]
         s_b = scores[idx]
         if len(np.unique(y_b)) < 2:
             continue
         try:
             # 阈值使用自助样本上的Youden选择，保持与主流程一致
             fpr, tpr, thresholds = roc_curve(y_b, s_b)
             thr_b = thresholds[np.argmax(tpr - fpr)]
             m = compute_binary_metrics(y_b, s_b, thr_b)
             acc.append(m["accuracy"])
             prec.append(m["precision"])
             rec.append(m["recall"])
             spec.append(m["specificity"])
             f1s.append(m["f1"])
             ppv.append(m["ppv"])
             npvs.append(m["npv"])
             youden.append(m["youden_index"])
             # PR-AUC
             pr.append(average_precision_score(y_b, s_b))
         except Exception:
             continue
     def ci(arr):
         if not arr:
             return (np.nan, np.nan)
         return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
     return {
         "accuracy": ci(acc),
         "precision": ci(prec),
         "recall": ci(rec),
         "specificity": ci(spec),
         "f1": ci(f1s),
         "pr_auc": ci(pr),
         "ppv": ci(ppv),
         "npv": ci(npvs),
         "youden_index": ci(youden),
     }


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    results = []
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            print(f"警告: 结局列不存在 -> {outcome}")
            continue

        # 仅保留0/1的样本
        y = df[outcome].copy()
        # 尝试将非整数二元标签转换为0/1
        unique_vals = sorted(pd.unique(y.dropna()))
        if len(unique_vals) > 2:
            print(f"跳过(非二分类): {outcome}, 唯一值={unique_vals}")
            continue
        if len(unique_vals) == 1:
            print(f"跳过(仅单一类别): {outcome}, 唯一值={unique_vals}")
            continue

        # 将标签规范为0/1
        # 假设较大的值为阳性(1)，较小为阴性(0)
        mapping = {unique_vals[0]: 0, unique_vals[-1]: 1}
        y = y.map(mapping)

        for biomarker in BIOMARKERS:
            if biomarker not in df.columns:
                print(f"警告: 生物标志物列不存在 -> {biomarker}")
                continue

            sub = df[[outcome, biomarker]].dropna()
            if sub.empty:
                print(f"跳过(无有效样本): {outcome}-{biomarker}")
                continue

            y_sub = sub[outcome].map(mapping).values.astype(int)
            x_sub = sub[biomarker].values.astype(float)

            # ROC-AUC
            roc_auc = np.nan
            roc_auc_low = np.nan
            roc_auc_high = np.nan
            pr_auc = np.nan
            try:
                if len(np.unique(y_sub)) == 2:
                    roc_auc = roc_auc_score(y_sub, x_sub)
                    roc_auc_low, roc_auc_high = bootstrap_auc(y_sub, x_sub, n_bootstrap=2000, random_state=42)
                    pr_auc = average_precision_score(y_sub, x_sub)
            except Exception as e:
                print(f"计算ROC/PR-AUC出错 {outcome}-{biomarker}: {e}")

            # Youden阈值与基于该阈值的二分类指标
            thr = np.nan
            sensitivity = specificity = accuracy = youden_index = np.nan
            precision = recall = f1 = ppv = npv = np.nan
            try:
                if len(np.unique(y_sub)) == 2:
                    thr = compute_youden_threshold(y_sub, x_sub)
                    m = compute_binary_metrics(y_sub, x_sub, thr)
                    sensitivity = m["sensitivity"]
                    specificity = m["specificity"]
                    accuracy = m["accuracy"]
                    youden_index = m["youden_index"]
                    precision = m["precision"]
                    recall = m["recall"]
                    f1 = m["f1"]
                    ppv = m["ppv"]
                    npv = m["npv"]
                    # 指标CI(含PR-AUC)
                    metrics_ci = bootstrap_metrics(y_sub, x_sub, n_bootstrap=2000, random_state=42)
                    acc_ci_low, acc_ci_high = metrics_ci["accuracy"]
                    prec_ci_low, prec_ci_high = metrics_ci["precision"]
                    rec_ci_low, rec_ci_high = metrics_ci["recall"]
                    spec_ci_low, spec_ci_high = metrics_ci["specificity"]
                    f1_ci_low, f1_ci_high = metrics_ci["f1"]
                    pr_ci_low, pr_ci_high = metrics_ci["pr_auc"]
                    ppv_ci_low, ppv_ci_high = metrics_ci["ppv"]
                    npv_ci_low, npv_ci_high = metrics_ci["npv"]
                    youden_ci_low, youden_ci_high = metrics_ci["youden_index"]
                else:
                    acc_ci_low = acc_ci_high = prec_ci_low = prec_ci_high = np.nan
                    rec_ci_low = rec_ci_high = spec_ci_low = spec_ci_high = np.nan
                    f1_ci_low = f1_ci_high = pr_ci_low = pr_ci_high = np.nan
                    ppv_ci_low = ppv_ci_high = npv_ci_low = npv_ci_high = np.nan
                    youden_ci_low = youden_ci_high = np.nan
            except Exception as e:
                print(f"计算阈值/二分类指标出错 {outcome}-{biomarker}: {e}")

            results.append({
                "outcome": outcome,
                "biomarker": biomarker,
                "roc_auc": roc_auc,
                "roc_auc_ci_low": roc_auc_low,
                "roc_auc_ci_high": roc_auc_high,
                "pr_auc": pr_auc,
                "pr_auc_ci_low": pr_ci_low,
                "pr_auc_ci_high": pr_ci_high,
                "accuracy": accuracy,
                "accuracy_ci_low": acc_ci_low,
                "accuracy_ci_high": acc_ci_high,
                "precision": precision,
                "precision_ci_low": prec_ci_low,
                "precision_ci_high": prec_ci_high,
                "recall": recall,
                "recall_ci_low": rec_ci_low,
                "recall_ci_high": rec_ci_high,
                "specificity": specificity,
                "specificity_ci_low": spec_ci_low,
                "specificity_ci_high": spec_ci_high,
                "f1": f1,
                "f1_ci_low": f1_ci_low,
                "f1_ci_high": f1_ci_high,
                "ppv": ppv,
                "ppv_ci_low": ppv_ci_low,
                "ppv_ci_high": ppv_ci_high,
                "npv": npv,
                "npv_ci_low": npv_ci_low,
                "npv_ci_high": npv_ci_high,
                "sensitivity": sensitivity,
                "youden_index": youden_index,
                "youden_index_ci_low": youden_ci_low,
                "youden_index_ci_high": youden_ci_high,
                "youden_threshold": thr,
                "n_samples": len(sub),
            })

    out_df = pd.DataFrame(results)
    # 排序：结局、指标名
    out_df.sort_values(by=["outcome", "biomarker"], inplace=True)
    # 保留三位小数以便表格展示
    for col in [
        "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "precision", "precision_ci_low", "precision_ci_high",
        "recall", "recall_ci_low", "recall_ci_high",
        "specificity", "specificity_ci_low", "specificity_ci_high",
        "f1", "f1_ci_low", "f1_ci_high",
        "ppv", "ppv_ci_low", "ppv_ci_high",
        "npv", "npv_ci_low", "npv_ci_high",
        "sensitivity",
        "youden_index", "youden_index_ci_low", "youden_index_ci_high",
        "youden_threshold"
    ]:
        out_df[col] = out_df[col].round(3)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"已保存单变量指标表: {OUTPUT_PATH}")
    # 打印前几行以便预览
    print(out_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()