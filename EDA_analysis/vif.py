import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

raw_cols = ["IL6", "IL10", "TNFα", "CRP", "ACTH", "PTC"]

def _disp_name(s: str) -> str:
    t = str(s)
    t = t.replace('TNFalpha', 'TNF-α').replace('TNFα', 'TNF-α')
    t = t.replace('IL6', 'IL-6').replace('IL10', 'IL-10')
    return t

# 用 VAE 插补后的完整数据
df = pd.read_csv(r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\dataset\xueye.csv")
if "TNFα" not in df.columns and "TNFalpha" in df.columns:
    df = df.rename(columns={"TNFalpha": "TNFα"})

corr = df[raw_cols].corr(method="spearman")

plt.figure(figsize=(10,6), dpi=600)
cmap = sns.blend_palette(["#3a70af", "#7badd3", "#c0d6e4", "#aed9f5"], as_cmap=True)
ax = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    vmin=-1, vmax=1, center=0,
    cmap=cmap,
    square=True,
    cbar_kws={"label": "Spearman ρ"},
    annot_kws={"size": 15}
)
ax.tick_params(axis='x', rotation=0, labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_xticklabels([_disp_name(l.get_text()) for l in ax.get_xticklabels()])
ax.set_yticklabels([_disp_name(l.get_text()) for l in ax.get_yticklabels()])
if ax.collections:
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Spearman ρ", size=15)
plt.tight_layout()
plt.savefig(r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\vif\corr_raw6_spearman.png", dpi=600)

# 导出相关矩阵为 CSV
corr.to_csv(r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\vif\corr_raw6_spearman.csv", encoding="utf-8-sig")
