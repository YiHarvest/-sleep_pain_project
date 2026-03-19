import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rain_path = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\imputation_plots\raincloud_all_features_grid.png"
bar_path = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\imputation_plots\score_barplot.png"
heat_path = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\vif\corr_raw6_spearman.png"

rain_img = mpimg.imread(rain_path)
bar_img = mpimg.imread(bar_path)
heat_img = mpimg.imread(heat_path)

fig = plt.figure(figsize=(18, 16), dpi=600)
gs = fig.add_gridspec(2, 1, height_ratios=[10, 6])

ax_top = fig.add_subplot(gs[0])
ax_top.imshow(rain_img)
ax_top.axis("off")

gs_bottom = gs[1].subgridspec(1, 2, wspace=0.02)
ax_bl = fig.add_subplot(gs_bottom[0])
ax_br = fig.add_subplot(gs_bottom[1])
ax_bl.imshow(bar_img)
ax_br.imshow(heat_img)
ax_bl.axis("off")
ax_br.axis("off")

plt.tight_layout()
plt.savefig(r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\EDA_analysis\group_plot.png", dpi=600)
plt.close()
