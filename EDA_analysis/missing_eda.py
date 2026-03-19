import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.stats import gaussian_kde
try:
    import missingno as msno
    HAS_MSNO = True
except ModuleNotFoundError:
    HAS_MSNO = False
import os

# 设置风格（与你 competition.py 中小提琴图一致）
# 风格：白色网格 + talk 语境；全局字体设置见下。
sns.set_theme(style="whitegrid", context="talk")
# 字体：标题 18.0，轴标签 18.0，刻度 16.5，图例 16.5
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 18.0   # <--- [手动调整] 此处控制全局标题大小（如 dist_ACTH_mirror.png 的标题）
plt.rcParams['axes.labelsize'] = 18.0   # <--- [手动调整] 此处控制全局轴标签大小（如 dist_ACTH_mirror.png 的 xy 轴名称）
plt.rcParams['xtick.labelsize'] = 18.0  # <--- [手动调整] 此处控制全局 X 轴刻度标签大小
plt.rcParams['ytick.labelsize'] = 16.5  # <--- [手动调整] 此处控制全局 Y 轴刻度标签大小
plt.rcParams['legend.fontsize'] = 16.5  # <--- [手动调整] 此处控制全局图例大小
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'

# ============================
# 用户配置
# ============================
INPUT_PATH = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\dataset\sleep_15_20%缺失值.csv"                # <-- sleep 数据路径
OUTPUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\eda_missing"              # 输出文件夹
TARGET = "Chronic_pain"

BASE_FEATURES = ["IL6", "IL10", "TNFα", "CRP", "ACTH", "PTC"]

def _disp_name(s: str) -> str:
    t = str(s)
    t = t.replace('TNFalpha', 'TNF-α').replace('TNFα', 'TNF-α')
    t = t.replace('IL6', 'IL-6').replace('IL10', 'IL-10')
    return t

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# 读取数据
# ============================
df = pd.read_csv(INPUT_PATH)
if 'TNFalpha' in df.columns:
    df.rename(columns={'TNFalpha': 'TNFα'}, inplace=True)

print("数据维度：", df.shape)
print("\n>>> 分析 6 个基础血液指标的缺失情况...\n")

sub_df = df[BASE_FEATURES + [TARGET]].copy()

# ============================
# 1. 缺失率表格
# ============================
missing_rate = sub_df[BASE_FEATURES].isnull().mean().sort_values(ascending=False)
missing_rate.to_csv(f"{OUTPUT_DIR}/missing_rate.csv")
print("缺失率：\n", missing_rate)

# 缺失率柱图尺寸：8x5 英寸；保存 dpi=600
plt.figure(figsize=(8,5))
sns.barplot(x=missing_rate.values, y=missing_rate.index, palette="Blues_r")
plt.title("缺失率（基础血液指标）")
plt.xlabel("Missing Rate")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/missing_rate_bar.png", dpi=600)
plt.close()

# ============================
# 2. Missingno 矩阵图
# ============================
# Missing 矩阵图尺寸：10x5 英寸；保存 dpi=300
plt.figure(figsize=(10,5))
if HAS_MSNO:
    msno.matrix(sub_df[BASE_FEATURES])
else:
    mat = sub_df[BASE_FEATURES].isnull().astype(int)
    sns.heatmap(mat.T, cmap=[[1,1,1],[0.77,0.31,0.32]], cbar=False)
plt.title("Missing Matrix")
plt.savefig(f"{OUTPUT_DIR}/missing_matrix.png", dpi=300)
plt.close()

# ============================
# 3. Missingno 共现图 (heatmap)
# ============================
# 缺失共现热力图尺寸：10x6 英寸；保存 dpi=600
plt.figure(figsize=(10,6))
miss_corr = sub_df[BASE_FEATURES].isnull().astype(int).corr()
# 配色：与镜像密度图一致的低饱和蓝色发散色盘
coheatmap_cmap = sns.blend_palette(["#8FBBD9", "#FFFFFF", "#9FD4E0"], as_cmap=True)
ax = sns.heatmap(
    miss_corr,
    cmap=coheatmap_cmap,
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 24.0}
)
ax.tick_params(axis='x', rotation=0, labelsize=24.0)
ax.tick_params(axis='y', labelsize=20.0)
ax.set_xticklabels([_disp_name(t.get_text()) for t in ax.get_xticklabels()], fontsize=24.0)
ax.set_yticklabels([_disp_name(t.get_text()) for t in ax.get_yticklabels()], fontsize=20.0)
cb = ax.collections[0].colorbar
cb.ax.tick_params(labelsize=24.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/missing_coheatmap.png", dpi=600)
plt.savefig(f"{OUTPUT_DIR}/missing_coheatmap.tif", dpi=600)
plt.close()

# ============================
# 4. 缺失与 Chronic_pain 的关联
# ============================
assoc_df = {}

for col in BASE_FEATURES:
    tmp = df[[col, TARGET]].copy()
    tmp[col + "_missing"] = tmp[col].isnull().astype(int)
    assoc = tmp.groupby(TARGET)[col + "_missing"].mean()
    assoc_df[col] = assoc

assoc_df = pd.DataFrame(assoc_df).T
assoc_df.columns = ["No Pain", "Pain"]

assoc_df.to_csv(f"{OUTPUT_DIR}/missing_vs_pain.csv")

sns.set_theme(style="white", context="talk")
palette = sns.diverging_palette(240, 20, s=60, l=70, n=3)
colors = [palette[0], palette[-1]]
# 缺失 vs 疼痛 柱图尺寸：6.75x4.0 英寸；保存 dpi=600；双端发散色（与相关热图一致）
ax = assoc_df.plot(kind="bar", figsize=(6.75, 4.0), width=0.95, color=colors, alpha=0.85)
# 放大 1.5 倍：原 7 → 10.5
ax.tick_params(axis='x', rotation=0, labelsize=10.5, colors="black")
ax.tick_params(axis='y', labelsize=10.5, colors="black")
plt.ylabel("Missing Rate", fontsize=10.5, color="black")
leg = ax.legend(fontsize=10.5)
leg.set_bbox_to_anchor((0.0, 1.0))
leg.set_loc("upper left")
sns.despine(ax=ax, top=True, right=True)
labels = [t.get_text() for t in ax.get_xticklabels()]
labels = [_disp_name(lbl) for lbl in labels]
ax.set_xticklabels(labels, rotation=0)
sns.despine(ax=ax, top=True, right=True)
ax.spines['left'].set_color('#BBBBBB')
ax.spines['bottom'].set_color('#BBBBBB')
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/missing_vs_pain_bar.png", dpi=600)
plt.savefig(f"{OUTPUT_DIR}/missing_vs_pain_bar.tif", dpi=600)
plt.close()

# ============================
# 5. 基础指标分布图（模仿小提琴色系）
# ============================
palette = {
    "IL6": "#8FBBD9",
    "IL10": "#91D0A5",
    "TNFα": "#F7B7A3",
    "CRP": "#E4D28A",
    "ACTH": "#E08C83",
    "PTC": "#9FD4E0"
}

for col in BASE_FEATURES:
    # 单特征分布图尺寸：8x5 英寸；保存 dpi=300
    plt.figure(figsize=(8,5))
    sns.kdeplot(df[col].dropna(), fill=True, color=palette[col], alpha=0.7, linewidth=0.7)
    sns.boxplot(x=df[col], color=palette[col], width=0.3, linewidth=0.7)
    plt.title(f"Distribution of {_disp_name(col)}", fontsize=18.0)
    ax = plt.gca()
    ymax = ax.get_ylim()[1]
    ytop = np.ceil(ymax*10)/10.0
    ax.set_ylim(-0.2, ytop if ytop > 0 else 0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dist_{col}.png", dpi=300)
    plt.close()

# 所有基础指标分布（含 box）尺寸：15x10 英寸；保存 dpi=600
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(BASE_FEATURES):
    ax = axes[i]
    sns.kdeplot(df[col].dropna(), fill=True, color=palette[col], alpha=0.7, ax=ax, linewidth=0.7)
    sns.boxplot(x=df[col], color=palette[col], width=0.3, ax=ax, linewidth=0.7)
    ax.set_title(f"Distribution of {_disp_name(col)}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ymax = ax.get_ylim()[1]
    ytop = np.ceil(ymax*10)/10.0
    ax.set_ylim(-0.2, ytop if ytop > 0 else 0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=palette[c], lw=3) for c in BASE_FEATURES]
fig.legend(handles, [_disp_name(c) for c in BASE_FEATURES], loc="lower center", ncol=len(BASE_FEATURES), frameon=False)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{OUTPUT_DIR}/dist_all_blood_with_box.png", dpi=600)
plt.close()
# 所有基础指标密度线图尺寸：10x6 英寸；保存 dpi=600
plt.figure(figsize=(10,6))
for col in BASE_FEATURES:
    sns.kdeplot(df[col].dropna(), color=palette[col], linewidth=0.7)
plt.xlabel("Value")
plt.ylabel("Density")
ax = plt.gca()
ymax = ax.get_ylim()[1]
ytop = np.ceil(ymax*10)/10.0
ticks = np.arange(0.0, ytop + 1e-8, 0.1)
plt.yticks(ticks)
plt.legend([_disp_name(c) for c in BASE_FEATURES], loc="best")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dist_all_blood.png", dpi=600)
plt.close()

for col in BASE_FEATURES:
    x = df[col].dropna().values
    if len(x) < 2:
        continue
    xs = np.linspace(np.min(x), np.max(x), 400)
    kde = gaussian_kde(x)
    ys = kde(xs)
    # 单指标镜像密度图尺寸：8x5 英寸；保存 dpi=600
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="white", context="talk")
    ax.fill_between(xs, 0, ys, color=palette[col], alpha=0.55)
    ax.fill_between(xs, 0, -ys, color=palette[col], alpha=0.55)
    jitter_amp = float(np.max(ys)) * 0.05
    n = len(x)
    k = min(n, 250)
    idx = np.random.choice(n, size=k, replace=False)
    x_s = x[idx]
    y_s = np.random.uniform(-jitter_amp, jitter_amp, size=k)
    ax.scatter(x_s, y_s, s=12, alpha=0.6, c=palette[col], edgecolors="white", linewidths=0.5, zorder=2)
    ax.axhline(0, color="#888", linewidth=1)
    ax.set_xlabel(_disp_name(col), fontsize=27.0)
    ax.set_ylabel("Density", fontsize=27.0)
    ax.set_title(f"Distribution of {_disp_name(col)}", fontsize=27.0)
    ax.tick_params(axis='x', labelsize=24.75)
    ax.tick_params(axis='y', labelsize=24.75)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/dist_{col}_mirror.png", dpi=600)
    plt.close()

# 所有基础指标镜像密度总图尺寸：15x10 英寸；保存 dpi=600
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(BASE_FEATURES):
    ax = axes[i]
    x = df[col].dropna().values
    if len(x) < 2:
        fig.delaxes(ax)
        continue
    xs = np.linspace(np.min(x), np.max(x), 400)
    kde = gaussian_kde(x)
    ys = kde(xs)
    ax.fill_between(xs, 0, ys, color=palette[col], alpha=0.55)
    ax.fill_between(xs, 0, -ys, color=palette[col], alpha=0.55)
    jitter_amp = float(np.max(ys)) * 0.05
    n = len(x)
    k = min(n, 250)
    idx = np.random.choice(n, size=k, replace=False)
    x_s = x[idx]
    y_s = np.random.uniform(-jitter_amp, jitter_amp, size=k)
    ax.scatter(x_s, y_s, s=12, alpha=0.6, c=palette[col], edgecolors="white", linewidths=0.5, zorder=2)
    ax.axhline(0, color="#888", linewidth=1)
    ax.set_title(f"{_disp_name(col)}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    sns.despine(ax=ax, left=True, bottom=False)
for j in range(len(BASE_FEATURES), len(axes)):
    fig.delaxes(axes[j])
letters = list("ABCDEF")
for i, ax in enumerate(axes[:len(BASE_FEATURES)]):
    ax.text(-0.07, 1.12, letters[i], transform=ax.transAxes, fontsize=24, fontweight="bold", va="top", ha="left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dist_all_blood_mirror.png", dpi=600)
plt.savefig(f"{OUTPUT_DIR}/dist_all_blood_mirror.tif", dpi=600)
plt.close()
print("\n>>> 缺失值 EDA 完成！所有图已保存到：", OUTPUT_DIR)
