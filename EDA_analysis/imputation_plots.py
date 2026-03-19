import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# ===== 新增：VAE 相关依赖 =====
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'ieee'])
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from scipy.stats import ks_2samp, wasserstein_distance, gaussian_kde

# ================================
# 设置风格（与小提琴图完全一致）
# ================================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 18.0
plt.rcParams['axes.labelsize'] = 18.0
plt.rcParams['xtick.labelsize'] = 16.5
plt.rcParams['ytick.labelsize'] = 16.5
plt.rcParams['legend.fontsize'] = 16.5
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'

# ================================
# 配置
# ================================
INPUT = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\dataset\sleep_15_20%缺失值.csv"
OUTDIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\imputation_plots"
os.makedirs(OUTDIR, exist_ok=True)

BASE = ["IL6", "IL10", "TNFα", "CRP", "ACTH", "PTC"]

def _disp_name(s: str) -> str:
    t = str(s)
    t = t.replace('TNFalpha', 'TNF-α').replace('TNFα', 'TNF-α')
    t = t.replace('IL6', 'IL-6').replace('IL10', 'IL-10')
    return t
RATIOS = {
    "IL6/IL10": ("IL6", "IL10"),
    "TNFα/IL10": ("TNFα", "IL10"),
    "CRP/IL10": ("CRP", "IL10"),
    "PTC/ACTH": ("PTC", "ACTH"),
    "PTC/IL6": ("PTC", "IL6"),
    "PTC/CRP": ("PTC", "CRP"),
    "IL6/TNFα": ("IL6", "TNFα"),
    "CRP/IL6": ("CRP", "IL6"),
    "ACTH/IL6": ("ACTH", "IL6"),
}

COLOR_MAP = {
    "Original": "#aed9f5",
    "MICE": "#c0d6e4",
    "Mean": "#7badd3",
    "KNN": "#aed9f5",
    "VAE": "#3a70af",
    
}

FEATURE_PALETTE = {
    "IL6": "#8FBBD9",
    "IL10": "#91D0A5",
    "TNFα": "#F7B7A3",
    "CRP": "#E4D28A",
    "ACTH": "#E08C83",
    "PTC": "#9FD4E0",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================================
# 读取数据
# ================================
df = pd.read_csv(INPUT)
df.columns = [c.replace('TNFalpha', 'TNFα') for c in df.columns]
df_base = df[BASE]

# ================================
# 传统插补方法
# ================================
def im_mice(df_in):
    imp = IterativeImputer(random_state=42, max_iter=40)
    return pd.DataFrame(imp.fit_transform(df_in), columns=df_in.columns, index=df_in.index)


def im_mean(df_in):
    imp = SimpleImputer(strategy="mean")
    return pd.DataFrame(imp.fit_transform(df_in), columns=df_in.columns, index=df_in.index)


def im_knn(df_in):
    imp = KNNImputer(n_neighbors=5)
    return pd.DataFrame(imp.fit_transform(df_in), columns=df_in.columns, index=df_in.index)

# ================================
# 新增：Masked VAE 模型定义
# ================================
class MaskedVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim=8, hidden_dim=64):
        super().__init__()
        input_dim = feature_dim * 2  # x + mask

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def encode(self, x, m):
        h = self.encoder(torch.cat([x, m], dim=-1))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, m):
        mu, logvar = self.encode(x, m)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, m, mu, logvar, beta=0.1):
    # 只在观测值位置计算重构损失
    recon_term = ((x_hat - x) ** 2 * m).sum()
    num_observed = m.sum()
    recon_loss = recon_term / (num_observed + 1e-8)

    # KL 散度
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / x.shape[0]

    return recon_loss + beta * kl, recon_loss.detach(), kl.detach()


def im_vae(df_in,
           num_epochs=200,
           batch_size=32,
           latent_dim=8,
           hidden_dim=64,
           lr=1e-3,
           beta=0.1,
           device=DEVICE):
    """
    对 BASE 六个指标做 Mask-aware VAE 插补
    """
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    x = df_in.values.astype(np.float32)        # (N, d)
    mask = ~np.isnan(x)                        # True = 观测
    mask_float = mask.astype(np.float32)

    # 列均值和方差（只用观测值）
    col_means = np.nanmean(x, axis=0)
    col_stds = np.nanstd(x, axis=0)
    # 处理全缺失或方差为 0 的情况
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    col_stds = np.where((np.isnan(col_stds)) | (col_stds == 0), 1.0, col_stds)

    # 标准化，缺失位置先填 0（均值）
    x_norm = (x - col_means) / col_stds
    x_norm[~mask] = 0.0

    # 构建 DataLoader
    tensor_x = torch.from_numpy(x_norm)
    tensor_m = torch.from_numpy(mask_float)
    dataset = TensorDataset(tensor_x, tensor_m)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    feature_dim = x.shape[1]
    model = MaskedVAE(feature_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n--- 训练 VAE 插补模型 ---")
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for xb, mb in loader:
            xb = xb.to(device)
            mb = mb.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(xb, mb)
            loss, recon_loss, kl_loss = vae_loss(x_hat, xb, mb, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_kl += kl_loss.item() * bs

        if epoch % 50 == 0 or epoch == 1 or epoch == num_epochs:
            avg_loss = total_loss / x.shape[0]
            avg_recon = total_recon / x.shape[0]
            avg_kl = total_kl / x.shape[0]
            print(f"Epoch {epoch:03d}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kl={avg_kl:.4f}")

    # 推理阶段：用 mu 作为潜在表示，进行确定性插补
    model.eval()
    with torch.no_grad():
        X_tensor = tensor_x.to(device)
        M_tensor = tensor_m.to(device)
        mu_all, logvar_all = model.encode(X_tensor, M_tensor)
        z_all = mu_all  # 使用后验均值
        X_hat_norm = model.decode(z_all).cpu().numpy()

    # 反标准化回原始量纲
    X_hat = X_hat_norm * col_stds + col_means

    # 仅在原缺失位置用 VAE 结果填补
    X_imputed = x.copy()
    X_imputed[~mask] = X_hat[~mask]

    df_vae = pd.DataFrame(X_imputed, columns=df_in.columns, index=df_in.index)
    return df_vae

# ================================
# 汇总所有插补方法（含 VAE）
# ================================
methods = {
    "MICE": im_mice(df_base),
    "Mean": im_mean(df_base),
    "KNN": im_knn(df_base),
    "VAE": im_vae(df_base),
}

# ================================
# 添加比例特征
# ================================
def add_ratios(df_imp):
    out = df_imp.copy()
    for r, (a, b) in RATIOS.items():
        out[r] = out[a] / out[b]
    return out


for m in methods:
    methods[m] = add_ratios(methods[m])

# ================================
# 评分指标（KS、Wasserstein、MeanShift）
# ================================
def compare(original, imputed):
    orig = original.dropna()
    imp = imputed.dropna()
    if len(orig) < 5 or len(imp) < 5:
        return np.nan, np.nan, np.nan
    ks = ks_2samp(orig, imp).statistic
    wd = wasserstein_distance(orig, imp)
    shift = abs(orig.mean() - imp.mean()) / (abs(orig.mean()) + 1e-6)
    return ks, wd, shift


scores = {}
details = []

for name, imp in methods.items():
    vals = []
    for feat in BASE + list(RATIOS.keys()):
        ks, wd, shift = compare(df[feat], imp[feat])
        vals.append((ks, wd, shift))
        details.append([name, feat, ks, wd, shift])

    arr = np.array(vals, dtype=float)
    ks_n = arr[:, 0] / (np.nanmax(arr[:, 0]) + 1e-6)
    wd_n = arr[:, 1] / (np.nanmax(arr[:, 1]) + 1e-6)
    sh_n = arr[:, 2] / (np.nanmax(arr[:, 2]) + 1e-6)

    score = np.nanmean(0.4 * ks_n + 0.4 * wd_n + 0.2 * sh_n)
    scores[name] = score

details_df = pd.DataFrame(details, columns=["Method", "Feature", "KS", "Wasserstein", "MeanShift"])
details_df.to_csv(f"{OUTDIR}/distribution_details.csv", index=False)

# ================================
# 1. 评分柱状图
# ================================
score_df = pd.DataFrame({"Method": list(scores.keys()), "Score": list(scores.values())})
score_df = score_df.sort_values("Score")

plt.figure(figsize=(9, 6))
# 图尺寸: 9x6 英寸; 保存 dpi=600。字体: 标题 18.0, 轴标签 18.0, 刻度 16.5, 图例 16.5。
plt.title("Imputation Method Scores (Lower is Better)")
sns.barplot(
    data=score_df,
    x="Method",
    y="Score",
    palette=[COLOR_MAP[m] for m in score_df["Method"]],
    linewidth=0.7,
)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/score_barplot.png", dpi=600)
plt.close()

# ================================
# 2. 各特征 KDE 分布对比图
# ================================
for feat in BASE:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[feat].dropna(), label="Original", color=COLOR_MAP["Original"], linewidth=0.7)

    for m, imp in methods.items():
        sns.kdeplot(imp[feat], label=m, color=COLOR_MAP[m], linewidth=0.7)

    plt.title(f"Distribution Comparison: {_disp_name(feat)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{feat}_kde.png", dpi=300)
    plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# KDE 总图尺寸: 15x8 英寸; 保存 dpi=600。
axes = axes.flatten()
handles_labels_done = False
for i, feat in enumerate(BASE):
    ax = axes[i]
    sns.kdeplot(df[feat].dropna(), label="Original", color=COLOR_MAP["Original"], linewidth=0.7, ax=ax)
    for m, imp in methods.items():
        sns.kdeplot(imp[feat], label=m, color=COLOR_MAP[m], linewidth=0.7, ax=ax)
    ax.set_title(_disp_name(feat))
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
for j in range(len(BASE), len(axes)):
    fig.delaxes(axes[j])

fig.legend(
    [axes[0].lines[0]] + axes[0].lines[1:],
    ["Original"] + list(methods.keys()),
    loc="lower center",
    ncol=5,
    frameon=False,
    prop={"size": 16.5},
)
letters = list("ABCDEF")
for i, ax in enumerate(axes[:len(BASE)]):
    ax.text(-0.07, 1.12, letters[i], transform=ax.transAxes,
            fontsize=24, fontweight="bold", va="top", ha="left")
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(f"{OUTDIR}/kde_all_features_grid.png", dpi=600)
plt.savefig(f"{OUTDIR}/kde_all_features_grid.tif", dpi=600)
plt.close()

# ================================
# 3. 雨云图（Violin + Box + Dot）
# ================================
def plot_raincloud_style(df_original, methods_dict, features, outdir, color_map):
    long_data = []
    for feat in features:
        df_temp = pd.DataFrame({
            "Method": "Original",
            "Feature": feat,
            "Value": df_original[feat].dropna(),
        })
        long_data.append(df_temp)
    for m, imp_df in methods_dict.items():
        for feat in features:
            df_temp = pd.DataFrame({
                "Method": m,
                "Feature": feat,
                "Value": imp_df[feat],
            })
            long_data.append(df_temp)
    plot_df = pd.concat(long_data, ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # 雨云总图尺寸: 18x10 英寸; 保存 dpi=600。配色: 指定四色 + Original=VAE= #9FD4E0。
    axes = axes.flatten()
    methods_order = ["Original"] + list(methods_dict.keys())

    for i, feat in enumerate(features):
        ax = axes[i]
        feat_df = plot_df[plot_df["Feature"] == feat]

        max_n = 100
        feat_df = feat_df.groupby("Method", group_keys=False).apply(
            lambda d: d.sample(n=min(len(d), max_n), random_state=42)
        )

        sns.violinplot(
            data=feat_df,
            x="Method",
            y="Value",
            order=methods_order,
            hue="Method",
            palette=color_map,
            inner=None,
            linewidth=0.7,
            saturation=0.85,
            width=0.9,
            ax=ax,
            legend=False,
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        for art in ax.findobj(lambda a: isinstance(a, PolyCollection)):
            try:
                art.set_edgecolor("gray")
                art.set_linewidth(1.0)
            except Exception:
                pass

        sns.stripplot(
            data=feat_df,
            x="Method",
            y="Value",
            order=methods_order,
            hue="Method",
            palette=color_map,
            size=3,
            edgecolor="none",
            linewidth=0.0,
            jitter=0.08,
            alpha=0.6,
            dodge=False,
            ax=ax,
            legend=False,
            zorder=2,
        )

        sns.boxplot(
            data=feat_df,
            x="Method",
            y="Value",
            order=methods_order,
            hue="Method",
            palette=color_map,
            width=0.16,
            boxprops={"zorder": 3, "facecolor": "none", "edgecolor": "gray", "linewidth": 0.75},
            medianprops={"color": "#E08C83", "linewidth": 1.2, "solid_capstyle": "round"},
            capprops={"color": "gray", "linewidth": 0.75},
            whiskerprops={"color": "gray", "linewidth": 0.75},
            flierprops={"marker": "", "markersize": 0},
            showfliers=False,
            ax=ax,
        )
        leg3 = ax.get_legend()
        if leg3 is not None:
            leg3.remove()

        ax.set_title(feat, fontsize=27.0)  # 雨云图标题放大1.5倍（原18.0）
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=27.0)  # 雨云图 y 轴标签放大1.5倍（原18.0）
        ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
        ax.tick_params(axis='x', rotation=0, labelsize=20.75)  # 雨云图 x 轴刻度放大1.5倍（原16.5）
        ax.tick_params(axis='y', labelsize=24.75)  # 雨云图 y 轴刻度放大1.5倍（原16.5）

    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Imputation Distribution Comparison (Raincloud-Style: Violin + Box)", fontsize=27.0)  # 雨云总图标题放大1.5倍（原18.0）
    letters = list("ABCDEF")
    for i, ax in enumerate(axes[:len(features)]):
        ax.text(-0.07, 1.12, letters[i], transform=ax.transAxes,
                fontsize=36, fontweight="bold", va="top", ha="left")  # 放大1.5倍（原24）
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{outdir}/raincloud_all_features_grid.png", dpi=600)
    plt.savefig(f"{outdir}/raincloud_all_features_grid.tif", dpi=600)
    plt.close()


def plot_raincloud_single(df_original, methods_dict, feature, outdir, color_map):
    long_data = []
    df_temp = pd.DataFrame({
        "Method": "Original",
        "Feature": feature,
        "Value": df_original[feature].dropna(),
    })
    long_data.append(df_temp)
    for m, imp_df in methods_dict.items():
        df_temp = pd.DataFrame({
            "Method": m,
            "Feature": feature,
            "Value": imp_df[feature],
        })
        long_data.append(df_temp)
    plot_df = pd.concat(long_data, ignore_index=True)
    methods_order = ["Original"] + list(methods_dict.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    # 单特征雨云图尺寸: 10x6 英寸; 保存 dpi=600。配色同上。
    feat_df = plot_df[plot_df["Feature"] == feature]
    max_n = 100
    feat_df = feat_df.groupby("Method", group_keys=False).apply(
        lambda d: d.sample(n=min(len(d), max_n), random_state=42)
    )

    sns.violinplot(
        data=feat_df,
        x="Method",
        y="Value",
        order=methods_order,
        hue="Method",
        palette=color_map,
        inner=None,
        linewidth=0.7,
        saturation=0.85,
        width=0.9,
        ax=ax,
        legend=False,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    for art in ax.findobj(lambda a: isinstance(a, PolyCollection)):
        try:
            art.set_edgecolor("gray")
            art.set_linewidth(1.0)
        except Exception:
            pass

    sns.stripplot(
        data=feat_df,
        x="Method",
        y="Value",
        order=methods_order,
        hue="Method",
        palette=color_map,
        size=3,
        edgecolor="none",
        linewidth=0.0,
        jitter=0.08,
        alpha=0.6,
        dodge=False,
        ax=ax,
        legend=False,
        zorder=2,
    )

    sns.boxplot(
        data=feat_df,
        x="Method",
        y="Value",
        order=methods_order,
        hue="Method",
        palette=color_map,
        width=0.16,
        boxprops={"zorder": 3, "facecolor": "none", "edgecolor": "gray", "linewidth": 0.75},
        medianprops={"color": "#E08C83", "linewidth": 1.2, "solid_capstyle": "round"},
        capprops={"color": "gray", "linewidth": 0.75},
        whiskerprops={"color": "gray", "linewidth": 0.75},
        flierprops={"marker": "", "markersize": 0},
        showfliers=False,
        ax=ax,
    )
    leg3 = ax.get_legend()
    if leg3 is not None:
        leg3.remove()

    ax.set_title(_disp_name(feature), fontsize=27.0)  # 雨云图标题放大1.5倍（原18.0）
    ax.set_xlabel("")
    ax.set_ylabel("Value", fontsize=27.0)  # 雨云图 y 轴标签放大1.5倍（原18.0）
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    ax.tick_params(axis='x', rotation=0, labelsize=24.75)  # 雨云图 x 轴刻度放大1.5倍（原16.5）
    ax.tick_params(axis='y', labelsize=24.75)  # 雨云图 y 轴刻度放大1.5倍（原16.5）
    plt.tight_layout()
    plt.savefig(f"{outdir}/raincloud_{feature}.png", dpi=600)
    plt.close()

print("\n--- 运行雨云图风格绘图 ---")
plot_raincloud_style(df, methods, BASE, OUTDIR, COLOR_MAP)
print(f"雨云图风格的对比图已保存至：{OUTDIR}/raincloud_all_features_grid.png")

for feat in BASE:
    plot_raincloud_single(df, methods, feat, OUTDIR, COLOR_MAP)

# ================================
# 可选：半小提琴工具函数（原脚本已定义）
# ================================
def half_violin(ax, data, color, side="right", bw_adjust=1.0, alpha=0.7):
    kde = gaussian_kde(data, bw_method=bw_adjust)
    x = np.linspace(float(np.min(data)), float(np.max(data)), 200)
    y = kde(x)
    y = y / (np.max(y) + 1e-12) * 0.4
    if side == "right":
        ax.fill_betweenx(x, 0, y, color=color, alpha=alpha, zorder=1)
    else:
        ax.fill_betweenx(x, -y, 0, color=color, alpha=alpha, zorder=1)
    return ax

print("\n==============================")
print("图像已生成，目录：", OUTDIR)
print("评分如下：")
print(score_df)
print("最佳方法：", score_df.iloc[0]["Method"])
print("==============================")
