# -*- coding: utf-8 -*-
"""
plot_paper_tree.py

功能:
    基于 surrogate_tree_testset_convex_T.py 的逻辑，
    使用 dtreeviz 库生成发表级的精美决策树可视化 (SVG/PDF)。

特点:
    - 自动加载集成模型生成 Teacher Labels
    - 使用 dtreeviz 生成带有分布直方图和渐变色的树
    - 包含特征名美化 (例如: TNFalpha -> TNF-α)
"""

import os
import joblib
import numpy as np
import pandas as pd
import dtreeviz
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# 1. 配置路径 (与原文件保持一致)
# =====================================================================

TEST_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv"
MODEL_PKL = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\final_ensemble_convex_T.pkl"
OUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\surrogate_tree_visualization"

# 创建输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 目标特征 (15个)
BLOOD_15_FEATURES = [
    "IL6", "IL10", "TNFalpha", "CRP", "ACTH", "PTC",
    "IL6/IL10", "TNFalpha/IL10", "CRP/IL10",
    "PTC/ACTH", "PTC/IL6", "PTC/CRP",
    "IL6/TNFalpha", "CRP/IL6", "ACTH/IL6",
]

# 树参数
TREE_MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 3
F1_THRESHOLD = 0.04
RANDOM_STATE = 42

# =====================================================================
# 2. 必须重新定义集成模型类 (否则 joblib.load 会报错)
# =====================================================================

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

# =====================================================================
# 3. 数据准备与模型预测逻辑
# =====================================================================

def main():
    print(f"[INFO] Reading test data: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    
    # 1. 准备特征数据
    X_test_top = test_df[BLOOD_15_FEATURES].copy()
    
    # 为了论文美观，重命名 DataFrame 的列名为更科学的格式
    # 如果你的系统支持 LaTeX 渲染，可以使用 r"$TNF-\alpha$" 等，否则使用 unicode
    feature_rename_map = {
        "TNFalpha": "TNFα",
        "TNFalpha/IL10": "TNFα / IL10",
        "IL6/TNFalpha": "IL6 / TNFα",
        "IL6": "IL6",
        "IL10": "IL10"
        # 其他特征根据需要添加，dtreeviz 支持 UTF-8
    }
    
    X_test_display = X_test_top.rename(columns=feature_rename_map)
    feature_names_display = list(X_test_display.columns)

    # 2. 加载模型并生成 Teacher Labels
    print(f"[INFO] Loading model: {MODEL_PKL}")
    ensemble = joblib.load(MODEL_PKL)
    
    # 注意：预测时必须用原始列名的数据，不能用重命名后的
    X_test_for_model = test_df.drop(columns=["Chronic_pain"])
    proba_test = ensemble.predict_proba(X_test_for_model)[:, 1]
    
    # 生成 Teacher Labels
    y_test_teacher = (proba_test >= F1_THRESHOLD).astype(int)
    print(f"[INFO] Teacher labels generated with threshold {F1_THRESHOLD}")

    # 3. 训练替代树 (Surrogate Tree)
    # 注意：fit 的时候我们使用重命名后的 X_test_display，这样画图时自动显示漂亮的名字
    clf = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    clf.fit(X_test_display, y_test_teacher)

    # =====================================================================
    # 4. 使用 dtreeviz 绘图 (关键步骤)
    # =====================================================================
    print("[INFO] Generating dtreeviz visualization...")

    # 初始化 dtreeviz 模型适配器
    # 更接近小提琴图的 6 色主题（参考 seaborn default palette）
    six_palette = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']
    custom_colors = {
        'classes': {2: [six_palette[0], six_palette[2]]},  # 低风险蓝，高风险红
        'edge': six_palette[5],
        'split_line': six_palette[4],
        'title': '#333333',
        'axis_label': '#333333',
        'tick_label': '#333333',
        'scatter_edge': six_palette[5]
    }

    viz_model = dtreeviz.model(
        clf,
        X_train=X_test_display,
        y_train=y_test_teacher,
        target_name="Risk Class",
        feature_names=feature_names_display,
        class_names=["Low Risk", "High Risk"]
    )

    try:
        v = viz_model.view(
            scale=2.2,
            fontname='Arial',
            fancy=True,
            colors=custom_colors
        )
    except TypeError:
        v = viz_model.view(
            scale=2.2,
            fontname='Arial',
            fancy=True
        )
    
    # 保存为 SVG (矢量图，放入论文最佳)
    save_path_svg = os.path.join(OUT_DIR, "paper_tree_full_15features.svg")
    v.save(save_path_svg)
    print(f"[SUCCESS] Saved SVG to: {save_path_svg}")
    
    # 保存为 PNG (备用，高分辨率)
    # dtreeviz 的 save 方法根据后缀自动判断
    try:
        save_path_png = os.path.join(OUT_DIR, "paper_tree_viz.png")
        v.save(save_path_png) 
        print(f"[SUCCESS] Saved PNG to: {save_path_png}")
    except Exception as e:
        print(f"[WARN] PNG save failed (graphviz issue?): {e}")

    try:
        save_path_tif = os.path.join(OUT_DIR, "paper_tree_viz.tif")
        v.save(save_path_tif)
        print(f"[SUCCESS] Saved TIFF to: {save_path_tif}")
    except Exception as e:
        print(f"[WARN] TIFF save failed (graphviz issue?): {e}")

    # 4.2 (可选) 生成极简版树，只看路径
    # 如果树太大，有时候不需要看直方图，可以把 fancy=False, x_data=None 关掉部分细节
    # 但 dtreeviz 的精髓就在于直方图，建议保留。

if __name__ == "__main__":
    main()
