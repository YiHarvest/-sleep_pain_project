import os
import argparse
import joblib
import dtreeviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import shap
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    import cairosvg
except Exception:
    cairosvg = None
try:
    from PIL import Image
except Exception:
    Image = None

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 0.75
plt.rcParams['grid.linewidth'] = 0.4
plt.rcParams['grid.color'] = '#DDDDDD'
plt.rcParams['lines.antialiased'] = True

ARTIFACTS_DEFAULT = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\tree_analysis\surrogate_tree_artifacts.joblib"
OUT_DIR_DEFAULT = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\tree_plotter"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=ARTIFACTS_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--scale", type=float, default=4.4)
    ap.add_argument("--fontname", default="Arial")
    ap.add_argument("--fancy", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = joblib.load(args.artifacts)
    clf = data["clf"]
    X_train = data["X_train"]
    y_train = data["y_train"]
    feature_names = data["feature_names"]

    six_palette = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']
    custom_colors = {
        'classes': {2: [six_palette[0], six_palette[2]]},
        'edge': six_palette[5],
        'split_line': six_palette[4],
        'title': '#333333',
        'axis_label': '#333333',
        'tick_label': '#333333',
        'scatter_edge': six_palette[5]
    }

    viz_model = dtreeviz.model(
        clf,
        X_train=X_train,
        y_train=y_train,
        target_name="Risk Class",
        feature_names=feature_names,
        class_names=["Low Risk", "High Risk"]
    )

    try:
        v = viz_model.view(scale=args.scale, fontname=args.fontname, fancy=args.fancy, colors=custom_colors)
    except TypeError:
        v = viz_model.view(scale=args.scale, fontname=args.fontname, fancy=args.fancy)

    svg_path = os.path.join(args.out_dir, "paper_tree_full_15features.svg")
    v.save(svg_path)
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_txt = f.read()
        svg_txt = re.sub(r'(<text[^>]*style=")([^\"]*)"',
                         lambda m: m.group(1) + ("font-weight:bold; " + m.group(2) if "font-weight" not in m.group(2) else m.group(2)) + '"',
                         svg_txt)
        svg_txt = re.sub(r'(<text\b(?![^>]*style=)[^>]*?)>', r'\1 style="font-weight:bold">', svg_txt)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_txt)
    except Exception:
        pass

    try:
        tif_path = os.path.join(args.out_dir, "paper_tree_full_15features.tif")
        png_tmp = os.path.join(args.out_dir, "paper_tree_full_15features_tmp.png")
        if cairosvg is not None:
            cairosvg.svg2png(url=svg_path, write_to=png_tmp, output_width=None, output_height=None, dpi=300)
            if Image is not None:
                img = Image.open(png_tmp)
                img.save(tif_path, compression="tiff_deflate")
                try:
                    os.remove(png_tmp)
                except Exception:
                    pass
        elif Image is not None:
            pass
    except Exception:
        pass

    # 生成水平布局（LR）版本并输出 PNG
    try:
        try:
            v_lr = viz_model.view(scale=args.scale, fontname=args.fontname, fancy=args.fancy, colors=custom_colors, orientation='LR')
        except TypeError:
            v_lr = viz_model.view(scale=args.scale, fontname=args.fontname, fancy=args.fancy, orientation='LR')
        svg_lr_path = os.path.join(args.out_dir, "paper_tree_full_15features_lr.svg")
        v_lr.save(svg_lr_path)
        try:
            with open(svg_lr_path, "r", encoding="utf-8") as f:
                svg_txt_lr = f.read()
            svg_txt_lr = re.sub(r'(<text[^>]*style=")([^\"]*)"',
                                lambda m: m.group(1) + ("font-weight:bold; " + m.group(2) if "font-weight" not in m.group(2) else m.group(2)) + '"',
                                svg_txt_lr)
            svg_txt_lr = re.sub(r'(<text\b(?![^>]*style=)[^>]*?)>', r'\1 style="font-weight:bold">', svg_txt_lr)
            with open(svg_lr_path, "w", encoding="utf-8") as f:
                f.write(svg_txt_lr)
        except Exception:
            pass
        png_lr_path = os.path.join(args.out_dir, "paper_tree_full_15features_lr.png")
        if cairosvg is not None:
            cairosvg.svg2png(url=svg_lr_path, write_to=png_lr_path, output_width=None, output_height=None, dpi=300)
    except Exception:
        pass

    

    # ========= Panel C: Leaf coverage & chronic pain rate =========
    leaf_info = data.get("leaf_info", [])
    if len(leaf_info) > 0:
        counts = np.array([x["count"] for x in leaf_info])
        rates = np.array([x["pos_rate"] for x in leaf_info])

        # 使用路径的最后一条规则（更接近叶端）作为标签
        rules_simple = []
        for li in leaf_info:
            if len(li["rules"]) > 0:
                feat, thresh, op = li["rules"][-1]
                rules_simple.append(f"{feat} {op} {thresh:.2f}")
            else:
                rules_simple.append("")

        # 按覆盖人数排序，取 Top 6
        idx = np.argsort(-counts)
        counts = counts[idx]
        rates = rates[idx]
        rules_simple = [rules_simple[i] for i in idx]

        top_k = min(6, len(counts))
        counts = counts[:top_k]
        rates = rates[:top_k]
        rules_simple = rules_simple[:top_k]

        # 颜色按风险率编码，并添加 colorbar（改为蓝系）
        cmap = mcolors.LinearSegmentedColormap.from_list("CustomBlues", ["#ECF4FB", "#4c72b0"]) 
        rmin, rmax = float(rates.min()), float(rates.max())
        if rmax > rmin:
            norm = mcolors.Normalize(vmin=rmin, vmax=rmax)
            colors = cmap((rates - rmin) / (rmax - rmin))
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            colors = cmap(np.full_like(rates, 0.7))

        fig, ax = plt.subplots(figsize=(9, 10))
        y = np.arange(top_k)
        bars = ax.barh(y, counts, color=colors, edgecolor="#333333", linewidth=0.8)

        # 纵轴直接显示简化规则作为标签
        ax.set_yticks(y)
        ax.set_yticklabels(rules_simple, fontsize=21)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        ax.set_xlabel("Number of patients", fontsize=27)
        ax.set_ylabel("", fontsize=27)
        ax.set_title("Leaf coverage", fontsize=30)
        ax.tick_params(axis="x", labelsize=27)
        ax.tick_params(axis="y", labelsize=21)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, 40)
        ax.set_ylim(-0.5, top_k - 0.5)
        ax.invert_xaxis()

        # 条末端标注人数与风险率，以及覆盖占比
        total_n = int(len(data.get("X_train", [])))
        for i, (b, r, n) in enumerate(zip(bars, rates, counts)):
            cov_text = f"cov={n/total_n:.0%}" if total_n > 0 else "cov=?"
            xmax = max(ax.get_xlim())
            x_end = b.get_x() + b.get_width()
            pad_base = 8.5
            char_unit = 0.6
            pad = pad_base + (len(cov_text) * char_unit)
            xpos = min(x_end + pad, xmax - 0.4)
            ax.text(
                xpos,
                b.get_y() + b.get_height() / 2,
                cov_text,
                va="center",
                ha="left",
                fontsize=18,
                color="#333333",
                clip_on=False,
                zorder=5,
            )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="5%", pad=0.6)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Chronic pain rate", fontsize=21)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.tick_params(labelsize=27)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "leaf_coverage.png"), dpi=300)
        plt.close(fig)

    rule_scores = {}
    for info_item in leaf_info:
        path_rules = info_item["rules"]
        leaf_weight = info_item["count"]
        leaf_risk = info_item["pos_rate"]
        rule_text = " AND ".join([f"{f} {op} {t:.2f}" for f, t, op in path_rules])
        if rule_text.strip() == "":
            continue
        depth_weight = 1 / (len(path_rules) + 1)
        score = leaf_weight * leaf_risk * depth_weight
        rule_scores[rule_text] = rule_scores.get(rule_text, 0) + score

    sorted_rules = sorted(rule_scores.items(), key=lambda x: -x[1])[:6]
    if len(sorted_rules) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))

        def _wrap_rule_label(text, max_len=40):
            parts = text.split(" AND ")
            lines, cur = [], ""
            for p in parts:
                s = p if cur == "" else cur + " AND " + p
                if len(s) <= max_len:
                    cur = s
                else:
                    lines.append(cur)
                    cur = p
            if cur != "":
                lines.append(cur)
            return "\n".join(lines)

        labels = [_wrap_rule_label(r[0]) for r in sorted_rules]
        importance = np.array([r[1] for r in sorted_rules], dtype=float)
        imp_min, imp_max = float(importance.min()), float(importance.max())
        if imp_max > imp_min:
            norm_imp = (importance - imp_min) / (imp_max - imp_min)
        else:
            norm_imp = np.full_like(importance, 0.7)

        bar_color = "#24428A"
        bar_edge = "#1A2B55"
        x = np.linspace(0.0, 1.0, 300)
        sigma = 0.18
        base_curve = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))

        y_ticks = np.arange(len(labels))
        max_amp = 0.0
        for i, lab in enumerate(labels):
            y0 = y_ticks[i]
            amp = 0.35 + 0.75 * norm_imp[i]
            y_curve = y0 + amp * base_curve
            max_amp = max(max_amp, amp)
            ax.fill_between(x, y0, y_curve, color=bar_color, alpha=0.9, edgecolor=bar_edge, linewidth=1.2)
            ax.text(1.03, y0 + amp * 1.02, f"{importance[i]:.2f}", va="center", fontsize=13, color="#333333", clip_on=False)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(labels, fontsize=13)
        ax.set_xlim(0.0, 1.12)
        ax.set_ylim(-0.5, y_ticks[-1] + max_amp + 0.5)
        ax.invert_yaxis()
        ax.set_xlabel("")
        ax.set_title("Global Decision Rule Importance", fontsize=18)
        ax.grid(axis="x", alpha=0.2)

        ymin, ymax = ax.get_ylim()
        ylines = np.linspace(ymin, ymax, 6)[1:-1]
        for yline in ylines[:4]:
            ax.axhline(yline, color="#BBBBBB", linewidth=0.8, alpha=0.35, zorder=0)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "global_rule_importance.png"), dpi=300)
        plt.close(fig)

    

    print(svg_path)

    

if __name__ == "__main__":
    main()
