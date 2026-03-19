#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shap_plotter.py — SHAP 绘图脚本
功能：读取 shap_analysis.py 生成的 joblib 数据，绘制并保存各种 SHAP 图表。
结果保存至：outputs/shap
"""

import os, sys, glob, argparse, warnings, math
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, PowerNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 600})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'
RGB_HC = LinearSegmentedColormap.from_list(
    "rgb_hc",
    ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"],
)
try:
    import matplotlib.cm as _cm
    _cm.register_cmap(name="rgb_hc", cmap=RGB_HC)
except Exception:
    pass
try:
    import matplotlib.cm as _cm
    _cm.register_cmap(name="rgb_hc", cmap=RGB_HC)
except Exception:
    pass
try:
    import shap as _sh
    _sh.plots.colors.red_blue = RGB_HC
    try:
        _sh.plots.colors.red = "#b2182b"
        _sh.plots.colors.blue = "#2166ac"
    except Exception:
        pass
except Exception:
    pass

FIGSIZE_SINGLE = (6, 4)
FIGSIZE_COMBINED = (10, 6)
FIGSIZE_DEPENDENCE = (7.08, 4.90)

BLUES_TRUNC = RGB_HC
BLUE_PINK = RGB_HC

INPUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap_analysis"
OUTPUT_DIR = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\shap_plotter"

def waterfall_colors(values):
    """
    使用 Version A 的 RGB_HC 给 waterfall 图着色。
    values: SHAP waterfall 中的特征贡献值数组
    """
    import numpy as np
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=np.min(values), vmax=np.max(values))
    return [RGB_HC(norm(v)) for v in values]

def plot_sample_topk_bar(exp_row, out_png, topk=15, title=None):
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        vals = np.array(getattr(exp_row, "values", exp_row), dtype=float).ravel()
    except Exception:
        vals = np.array(exp_row, dtype=float).ravel()
    n = vals.shape[0]
    if n == 0:
        fig, ax = plt.subplots(figsize=FIGSIZE_DEPENDENCE)
        if title:
            ax.set_title(title)
        plt.tight_layout(); plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close()
        return
    if hasattr(exp_row, "feature_names"):
        names_raw = list(exp_row.feature_names)
    else:
        names_raw = []
    if len(names_raw) != n:
        names_raw = [f"f{i}" for i in range(n)]
    mask = np.abs(vals) > 0
    vals = vals[mask]
    names_raw = np.array(names_raw)[mask].tolist()
    if vals.size == 0:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        if title:
            ax.set_title(title)
        plt.tight_layout(); plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close()
        return
    order = np.argsort(np.abs(vals))[::-1][:topk]
    vals = vals[order]
    names = [_disp_name(n) for n in np.array(names_raw)[order].tolist()]
    y = np.arange(vals.size)
    RED = "#b2182b"; BLUE = "#2166ac"
    colors = [RED if v > 0 else BLUE for v in vals]
    try:
        import os as _os
        fn = _os.path.basename(str(out_png)).lower()
        is_tn = ("waterfall_tn" in fn)
    except Exception:
        is_tn = False
    plot_vals = np.abs(vals) if is_tn else vals
    fig, ax = plt.subplots(figsize=FIGSIZE_DEPENDENCE)
    ax.barh(y, plot_vals, color=colors, edgecolor="black", linewidth=0.6, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=19)
    ax.tick_params(axis='y', labelsize=19)
    ax.yaxis.tick_left()
    try:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    except Exception:
        pass
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    try:
        ax.xaxis.tick_bottom()
        ax.tick_params(axis='x', labelsize=19, top=False, bottom=True)
    except Exception:
        ax.tick_params(axis='x', labelsize=19)
    if title:
        ax.set_title(title, fontsize=19)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close()

def _recolor_waterfall_ax(ax, exp_row, exp_all=None):
    try:
        # 固定色：正贡献红色、负贡献蓝色
        RED = "#b2182b"
        BLUE = "#2166ac"
        # 直接依据条的水平位置（x 方向）判定颜色，覆盖所有 patch 类型
        for p in ax.patches:
            try:
                bb = p.get_extents()
                x0, x1 = float(bb.x0), float(bb.x1)
                if np.isfinite(x0) and np.isfinite(x1) and not np.isclose(x0, x1):
                    cx = (x0 + x1) / 2.0
                    col = RED if cx > 0 else BLUE
                    p.set_facecolor(col)
                    try:
                        p.set_edgecolor(col)
                        p.set_alpha(1.0)
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception:
        pass

def _disp_name(s: str) -> str:
    t = str(s)
    t = t.replace("TNFalpha", "TNF-α")
    t = t.replace("TNFα", "TNF-α")
    t = t.replace("IL6", "IL-6")
    t = t.replace("IL10", "IL-10")
    t = t.replace("PTC", "Cortisol")
    return t

def replace_tnfalpha(names):
    return [_disp_name(n) for n in names]

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def _compute_topk_for_exp(exp, topk: int):
    mean_abs = np.abs(exp.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:topk]
    top_vals = mean_abs[order]
    top_names = np.array(exp.feature_names)[order]
    return order, top_vals, top_names

def _scatter_beeswarm_axes(ax, shap_values, data_values, feature_names, top_idx, cmap="rgb_hc", random_state: int = 42):
    rng = np.random.default_rng(random_state)
    norm = PowerNorm(gamma=0.6, vmin=0, vmax=1)
    feature_labels = replace_tnfalpha([feature_names[i] for i in top_idx])
    y_positions = np.arange(len(feature_labels))
    for pos, idx in enumerate(top_idx):
        shap_col = shap_values[:, idx]
        feat_col = data_values[:, idx].astype(float)
        if np.allclose(feat_col.max(), feat_col.min()):
            feat_norm = np.full_like(feat_col, 0.5, dtype=float)
        else:
            feat_norm = (feat_col - feat_col.min()) / (feat_col.max() - feat_col.min())
        order_samples = np.argsort(shap_col)
        shap_sorted = shap_col[order_samples]
        feat_sorted = feat_norm[order_samples]
        jitter = rng.uniform(-0.004, 0.004, size=shap_sorted.shape[0])
        y_jitter = pos + jitter
        ax.scatter(shap_sorted, y_jitter, c=RGB_HC(feat_sorted), s=18, alpha=0.85, linewidths=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_labels, fontsize=6)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP value", fontsize=7)
    ax.set_title("Sample-wise impact", fontsize=7)
    return norm

def plot_beeswarm(exp: shap.Explanation, out_png: str, topk: int = 20, max_samples: int = 1000, random_state: int = 42):
    shap_values = exp.values
    data_values = exp.data
    feature_names = list(exp.feature_names)
    idx = np.arange(shap_values.shape[0])
    if len(idx) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=max_samples, replace=False)
    shap_values = shap_values[idx, :]
    data_values = data_values[idx, :]
    
    # Use the order from plot_topk_bar (which matches original script logic usually)
    # But here we compute it locally
    top_idx, _, _ = _compute_topk_for_exp(exp, topk)
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    norm = _scatter_beeswarm_axes(ax, shap_values, data_values, feature_names, top_idx, cmap="rgb_hc", random_state=random_state)
    
    for sp in ["top","right","left"]:
        ax.spines[sp].set_visible(False)
    
    xmax = np.nanmax(np.abs(shap_values[:, top_idx]))
    if not np.isfinite(xmax) or xmax == 0:
        xmax = 0.01
    
    ax.set_xlim(-0.075, 0.075)
    ticks = np.linspace(-0.075, 0.075, 7)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.3f}" for t in ticks], fontsize=11)
    ax.tick_params(axis='y', labelsize=8)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.4)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=RGB_HC)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Normalized feature value", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    base, _ = os.path.splitext(out_png)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    print(f"[OK] Beeswarm saved: {out_png}")
    plt.close(fig)

def plot_topk_bar(exp: shap.Explanation, out_png: str, topk: int = 20):
    order, top_vals, top_names = _compute_topk_for_exp(exp, topk)
    top_names = replace_tnfalpha(top_names)
    
    # Match original script: largest at bottom?
    # Original script: 
    # plot_topk_bar(exp_all, ..., topk=args.topk_bar)
    # Let's assume standard logic: largest at top is default, but user requested largest at bottom.
    # In my analysis of original script, I didn't see the implementation of plot_topk_bar, 
    # but I saw render_shap_combined using ax_bar.invert_xaxis().
    
    # User request: "数值最大的在最下面" (Largest at bottom).
    # Also: "横轴范围固定为从右到左为 0.00000 到 0.03000"
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # To put largest at bottom (index 0), we plot them at y=0.
    # order is [Index of Largest, Index of 2nd Largest, ...]
    y_pos = np.arange(len(top_names))
    
    # Colors: Darker for larger values
    norm = Normalize(vmin=0, vmax=top_vals.max())
    colors = RGB_HC(norm(top_vals))
    
    ax.barh(y_pos, top_vals, color=colors, edgecolor="black", linewidth=0.6, height=0.6)
    
    # X-axis settings
    ax.set_xlim(0.0, 0.03)
    ax.invert_xaxis() # 0.03 (left) -> 0.0 (right)
    
    ticks = np.arange(0.0, 0.03001, 0.005)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.5f}" for t in ticks], fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)
    
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False) # Original script sets right visible? 
    # In shap_analysis_from_final_report.py: ax.spines['left'].set_visible(False), ax.spines['top'].set_visible(False).
    # Right spine? ax.yaxis.tick_right(). Usually Right spine is needed for ticks.
    # But render_shap_combined has ax_bar.spines["right"].set_visible(False).
    # plot_topk_bar in original: ax.spines['left'].set_visible(False), ax.spines['top'].set_visible(False).
    # Doesn't say about right. Default is visible.
    
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel("Mean |SHAP| on probability", fontsize=9.5)
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
    
    maxv = 0.03
    for y, v in zip(y_pos, top_vals):
        ax.text(v + maxv*0.02, y, f"{v:.4f}", va="center", ha="right", fontsize=10, color="black")
    
    plt.tight_layout()
    base, _ = os.path.splitext(out_png)
    fig.savefig(out_png, bbox_inches='tight', dpi=600)
    print(f"[OK] TopK bar saved: {out_png}")
    plt.close(fig)

def plot_dependences(exp: shap.Explanation, out_dir: str, topm: int = 8):
    vals = np.abs(exp.values).mean(axis=0)
    order = np.argsort(vals)[::-1][:topm]
    names = np.array(exp.feature_names)[order]
    try:
        pair_indices = {}
        for f in names:
            j_list = shap.approximate_interactions(f, exp.values, exp.data, feature_names=list(exp.feature_names))
            pair_indices[f] = j_list[1] if len(j_list) > 1 else None
    except Exception:
        pair_indices = {f: None for f in names}
    for f in names:
        fj = pair_indices.get(f, None)
        try:
            try:
                shap.plots.colors.red_blue = RGB_HC
                shap.plots.colors.blue_red = RGB_HC
            except Exception:
                pass
            if fj is not None and 0 <= fj < len(exp.feature_names):
                color_name = exp.feature_names[fj]
                shap.plots.scatter(exp[:, f], color=exp[:, color_name], show=False, cmap=RGB_HC, dot_size=64)
                plt.title("")  # 标题为空，未设置字体大小
                fn = os.path.join(out_dir, f"dependence_{f.replace('/', '_')}__color_{color_name.replace('/', '_')}.png")
            else:
                shap.plots.scatter(exp[:, f], show=False, cmap=RGB_HC, dot_size=64)
                plt.title("")  # 标题为空，未设置字体大小
                fn = os.path.join(out_dir, f"dependence_{f.replace('/', '_')}.png")
            ax = plt.gca()
            ax.tick_params(axis='x', labelsize=19)  # x轴刻度字体≈1.2×原始(16→19)
            ax.tick_params(axis='y', labelsize=19)  # y轴刻度字体与x轴一致(≈1.2×)
            xl = _fix_disp_name(ax.get_xlabel())
            yl = _fix_disp_name(ax.get_ylabel())
            ax.set_xlabel(xl, fontsize=19, fontweight='bold')  # x轴标签字体≈1.2×原始(16→19)
            ax.set_ylabel(yl, fontsize=19, fontweight='bold')  # y轴标签字体与x轴一致(≈1.2×)
            fig = ax.figure
            fig.subplots_adjust(bottom=0.15)  # 固定底部边距，增大字体时防止 x 轴上移
            for _ax in fig.axes:
                if _ax is ax:
                    continue
                cbar_lbl = _ax.get_ylabel()
                if isinstance(cbar_lbl, str) and len(cbar_lbl) > 0:
                    _ax.set_ylabel(_fix_disp_name(cbar_lbl), fontsize=12, fontweight='bold')  # 右侧色条标签字体保持12
                    _ax.tick_params(axis='y', labelsize=12)  # 右侧色条刻度字体保持12
            plt.savefig(fn, dpi=600, bbox_inches='tight'); print(f"[OK] Dependence saved: {fn}"); plt.close()
        except Exception:
            continue

def _fix_disp_name(n: str) -> str:
    s = _disp_name(n)
    s = s.replace("\n", " ")
    return s

def _spec_to_feat(name: str) -> str:
    return name.replace("_", "/")

def plot_dependences_specific(exp: shap.Explanation, out_dir: str, specs: list[tuple[str, str]]):
    feature_set = set(exp.feature_names)
    for base_spec, color_spec in specs:
        f = _spec_to_feat(base_spec)
        c = _spec_to_feat(color_spec)
        if f not in feature_set:
            continue
        try:
            try:
                shap.plots.colors.red_blue = RGB_HC
                shap.plots.colors.blue_red = RGB_HC
            except Exception:
                pass
            if c in feature_set:
                shap.plots.scatter(exp[:, f], color=exp[:, c], show=False, cmap=RGB_HC, dot_size=64)
                plt.title("")
                fn = os.path.join(out_dir, f"dependence_{base_spec}__color_{color_spec}.png")
            else:
                shap.plots.scatter(exp[:, f], show=False, cmap=RGB_HC, dot_size=64)
                plt.title("")
                fn = os.path.join(out_dir, f"dependence_{base_spec}.png")
            ax = plt.gca()
            ax.tick_params(axis='x', labelsize=19)  # x轴刻度字体≈1.2×原始(16→19)
            ax.tick_params(axis='y', labelsize=19)  # y轴刻度字体与x轴一致(≈1.2×)
            xl = _fix_disp_name(ax.get_xlabel())
            yl = _fix_disp_name(ax.get_ylabel())
            ax.set_xlabel(xl, fontsize=19, fontweight='bold')  # x轴标签字体≈1.2×原始(16→19)
            ax.set_ylabel(yl, fontsize=19, fontweight='bold')  # y轴标签字体与x轴一致(≈1.2×)
            fig = ax.figure
            fig.subplots_adjust(bottom=0.15)  # 固定底部边距，增大字体时防止 x 轴上移
            for _ax in fig.axes:
                if _ax is ax:
                    continue
                cbar_lbl = _ax.get_ylabel()
                if isinstance(cbar_lbl, str) and len(cbar_lbl) > 0:
                    _ax.set_ylabel(_fix_disp_name(cbar_lbl), fontsize=19, fontweight='bold')  # 右侧色条标签字体与x轴一致(≈1.2×)
                    _ax.tick_params(axis='y', labelsize=19)  # 右侧色条刻度字体与x轴一致(≈1.2×)
            plt.savefig(fn, dpi=600, bbox_inches='tight'); print(f"[OK] Dependence saved: {fn}"); plt.close()
        except Exception:
            continue

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

def plot_waterfalls(exp_all: shap.Explanation, oof_p: np.ndarray, oof_y: np.ndarray, out_dir: str):
    try:
        from sklearn.metrics import f1_score
    except Exception:
        f1_score = None
    if oof_p.shape[0] == 0 or oof_y.shape[0] == 0:
        return
    thr_candidates = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1 = -1.0
    for t in thr_candidates:
        yhat = (oof_p >= t).astype(int)
        try:
            f1 = f1_score(oof_y.astype(int), yhat)
        except Exception:
            f1 = (yhat==oof_y.astype(int)).mean()
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    picks = pick_samples_for_waterfall(oof_y.astype(int), oof_p, best_thr)
    for tag, idx in picks:
        try:
            try:
                shap.plots.colors.red_blue = RGB_HC
            except Exception:
                pass
            title = f"{tag.upper()}  y={int(oof_y[idx])}  p={oof_p[idx]:.3f}  thr(F1)={best_thr:.3f}"
            base_png = os.path.join(out_dir, f"waterfall_{tag}.png")
            plot_sample_topk_bar(exp_all[idx], base_png, topk=15, title=title)
            try:
                vals_all = np.array(getattr(exp_all[idx], "values", exp_all[idx]), dtype=float).ravel()
                base_val = float(np.ravel(getattr(exp_all[idx], "base_values", [0.0]))[0]) if hasattr(exp_all[idx], "base_values") else 0.0
                sum_all = float(vals_all.sum())
                pred_lbl = "pain" if float(oof_p[idx]) >= float(best_thr) else "normal"
                fig, ax = plt.subplots(figsize=FIGSIZE_DEPENDENCE)
                order = np.argsort(np.abs(vals_all))[::-1][:15]
                vals = vals_all[order]
                names = [_disp_name(n) for n in np.array(exp_all.feature_names)[order].tolist()]
                y_pos = np.arange(vals.size)
                cols = ["#b2182b" if v > 0 else "#2166ac" for v in vals]
                ax.barh(y_pos, vals, color=cols, edgecolor="black", linewidth=0.6, height=0.6)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, fontsize=19)
                ax.tick_params(axis='y', labelsize=19)
                ax.yaxis.tick_left()
                try:
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                except Exception:
                    pass
                ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
                ax.xaxis.tick_bottom()
                ax.tick_params(axis='x', labelsize=19, top=False, bottom=True)
                ax.set_title(title, fontsize=19)
                txt = f"Pred={pred_lbl}    y={int(oof_y[idx])}    p={float(oof_p[idx]):.3f}    thr={best_thr:.3f}\nbase={base_val:.4f}    sum={sum_all:.4f}    base+sum={base_val+sum_all:.4f}"
                plt.gcf().text(0.01, 0.98, txt, ha="left", va="top", fontsize=12, bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5))
                out_labeled = os.path.join(out_dir, f"waterfall_{tag}_label.png")
                plt.tight_layout()
                plt.savefig(out_labeled, dpi=600, bbox_inches='tight')
                plt.close()
                print(f"[OK] Waterfall labeled saved: {out_labeled}")
                try:
                    print(f"[INFO] {tag}: p={float(oof_p[idx]):.6f}, thr={best_thr:.6f}, base={base_val:.6f}, sum={sum_all:.6f}, base+sum={base_val+sum_all:.6f}, pred={pred_lbl}")
                except Exception:
                    pass
            except Exception:
                pass
            try:
                try:
                    shap.plots.colors.red_blue = RGB_HC
                except Exception:
                    pass
                shap.plots.waterfall(exp_all[idx], show=False)
                ax2 = plt.gca()
                ax2.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")
                plt.xticks(rotation=0)
                txt2 = f"Pred={'pain' if float(oof_p[idx]) >= float(best_thr) else 'normal'}    y={int(oof_y[idx])}    p={float(oof_p[idx]):.3f}    thr={best_thr:.3f}"
                plt.gcf().text(0.01, 0.98, txt2, ha="left", va="top", fontsize=12, bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5))
                out_waterfall_labeled = os.path.join(out_dir, f"waterfall_{tag}_waterfall_label.png")
                plt.tight_layout()
                plt.savefig(out_waterfall_labeled, dpi=600, bbox_inches='tight')
                plt.close()
                print(f"[OK] SHAP Waterfall labeled saved: {out_waterfall_labeled}")
            except Exception:
                pass
            print(f"[OK] Waterfall saved: {base_png}")
        except Exception:
            with open(os.path.join(out_dir, f'waterfall_{tag}.txt'), 'w', encoding='utf-8') as fw:
                fw.write(f"Sample index={idx}, y={int(oof_y[idx])}, p={oof_p[idx]:.4f}, thr={best_thr:.4f}\n")

def plot_waterfalls_force(exp_all: shap.Explanation, out_dir: str):
    try:
        vals = exp_all.values
        net = vals.sum(axis=1)
        idx_all = np.arange(vals.shape[0])
        tp = int(idx_all[np.argmax(net)]) if idx_all.size > 0 else None
        tn = int(idx_all[np.argmin(net)]) if idx_all.size > 0 else None
        border = int(idx_all[np.argmin(np.abs(net))]) if idx_all.size > 0 else None
        picks = [('tp', tp), ('tn', tn), ('borderline', border)]
        for tag, idx in picks:
            if idx is None:
                continue
            try:
                try:
                    shap.plots.colors.red_blue = RGB_HC
                except Exception:
                    pass
                title = f"{tag.upper()}  idx={idx}"
                base_png = os.path.join(out_dir, f"waterfall_{tag}.png")
                plot_sample_topk_bar(exp_all[idx], base_png, topk=15, title=title)
                print(f"[OK] Waterfall saved: {base_png}")
            except Exception:
                try:
                    with open(os.path.join(out_dir, f'waterfall_{tag}.txt'), 'w', encoding='utf-8') as fw:
                        fw.write(f"Sample index={idx}\n")
                except Exception:
                    pass
    except Exception:
        return

def plot_pathway_bar(out_dir, df_path_csv):
    if not os.path.exists(df_path_csv):
        print(f"[Warn] {df_path_csv} not found, skipping pathway bar.")
        return

    df_path = pd.read_csv(df_path_csv)

    if 'total_mean_abs_shap' in df_path.columns:
        df_path.sort_values('total_mean_abs_shap', ascending=False, inplace=True)

    groups = replace_tnfalpha(list(df_path['group']))
    values = np.array(df_path['total_mean_abs_shap'], dtype=float)
    y_pos = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    colors = RGB_HC(np.linspace(0.3, 0.8, len(values)))

    ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.75, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups, fontsize=14)
    ax.yaxis.tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel("Total mean |SHAP|", fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(alpha=0.3, linewidth=0.4, color="#DDDDDD")

    plt.tight_layout()
    path_png = os.path.join(out_dir, 'pathway_shap_bar.png')
    fig.savefig(path_png, dpi=600, bbox_inches='tight')
    print(f"[OK] Pathway bar saved: {path_png}")
    plt.close(fig)

def plot_shap_abs_hist(exp: shap.Explanation, out_png: str):
    vals = np.abs(exp.values).sum(axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.hist(vals, bins=50, color=RGB_HC(0.7), edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Sum of |SHAP| values", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.set_xlim(left=0)
    plt.tight_layout()
    base, _ = os.path.splitext(out_png)
    plt.savefig(out_png, dpi=600)
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    
    # Find joblib files
    files = glob.glob(os.path.join(INPUT_DIR, "shap_data_*.joblib"))
    if not files:
        print(f"No data found in {INPUT_DIR}")
        return
        
    for fpath in files:
        print(f"Processing {fpath}...")
        data = joblib.load(fpath)
        exp_all = data['exp_all']
        oof_p = data.get('oof_p', np.array([]))
        oof_y = data.get('oof_y', np.array([]))
        # data also contains 'df_path' (pathway summary CSV path) potentially, 
        # or we can find it in the same dir
        
        # Plot Beeswarm
        plot_beeswarm(exp_all, os.path.join(OUTPUT_DIR, 'beeswarm.png'))
        
        # Plot TopK Bar
        plot_topk_bar(exp_all, os.path.join(OUTPUT_DIR, 'top20_bar.png'), topk=20)
        
        # Plot Dependences (auto top-m + specified list)
        plot_dependences(exp_all, OUTPUT_DIR, topm=8)
        specs = [
            ("ACTH", "IL10"),
            ("ACTH_IL6", "IL6_IL10"),
            ("ACTH_IL6", "PTC_ACTH"),
            ("CRP", "IL10"),
            ("IL6_IL10", "IL6_TNFalpha"),
            ("IL10", "TNFalpha_IL10"),
            ("PTC", "ACTH"),
            ("PTC", "CRP_IL10"),
            ("PTC_ACTH", "ACTH"),
            ("PTC_ACTH", "IL10"),
            ("PTC_CRP", "IL6_TNFalpha"),
            ("PTC_IL6", "ACTH_IL6"),
            ("PTC_IL6", "PTC_ACTH"),
        ]
        plot_dependences_specific(exp_all, OUTPUT_DIR, specs)
        
        # Plot Pathway Bar
        # We need the CSV.
        # In shap_analysis.py, we saved 'pathway_shap_summary.csv' in output dir?
        # Let's assume it's in INPUT_DIR
        pathway_csv = os.path.join(INPUT_DIR, 'pathway_shap_summary.csv')
        plot_pathway_bar(OUTPUT_DIR, pathway_csv)
        
        
        
        # Plot Abs Hist
        plot_shap_abs_hist(exp_all, os.path.join(OUTPUT_DIR, 'shap_abs_hist.png'))
        plot_waterfalls(exp_all, oof_p, oof_y, OUTPUT_DIR)
        if oof_p.shape[0] == 0 or oof_y.shape[0] == 0:
            plot_waterfalls_force(exp_all, OUTPUT_DIR)

if __name__ == '__main__':
    main()
