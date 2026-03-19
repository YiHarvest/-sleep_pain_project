# 失眠患者疼痛分类任务（HemoPain）项目说明

本项目旨在基于血液生物标志物与部分临床变量，对慢性疼痛进行二分类预测，并提供稳健的模型评估与可视化。

![alt text](01.流程图第三版-最新版-cort.png)

## 快速开始
- 准备数据（CSV）：
  - 训练集：d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\train.csv
  - 测试集：d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\dataset\test.csv
- 运行核心脚本（建议依次执行）：
  - 基线模型与小提琴图：  
    ```bash
    python d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\competition.py
    ```
  - 集成学习与小提琴图、汇总：  
    ```bash
    python d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\stacking_competition.py
    ```
  - Bootstrap 评估（固定测试集上抽样置信区间与直方图）：  
    ```bash
    python d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\bootstrap_rank_platt.py
    ```

## 主要方法
- 基线模型（见 [competition.py](file:///d:/yiqy/sleepProjexts/发文章版_疼痛二分类任务/最终版本/competition.py)）
  - Logistic Regression、SVM、Random Forest、Gradient Boosting 等
  - 支持阈值策略与校准（isotonic/sigmoid），输出多项指标的 5 折 CV 分布与汇总
- 集成学习（见 [stacking_competition.py](file:///d:/yiqy/sleepProjexts/发文章版_疼痛二分类任务/最终版本/stacking_competition.py)）
  - HemoPain-Ensemble（logit 凸组合 + 温度缩放）
  - ENS-LogitT-AUC（基于 AUC 优化的 logit+T）
  - ENS-RankPlatt（秩平均 + Platt 标定）
  - ENS-Stack（两基学习器概率的 stacking LR）
  - 三类阈值策略：F1 最优、Recall≥0.80 下 PPV 最大、Youden 最优
- Bootstrap 评估（见 [bootstrap_rank_platt.py](file:///d:/yiqy/sleepProjexts/发文章版_疼痛二分类任务/最终版本/bootstrap_rank_platt.py)）
  - 在固定测试集上对最终集成模型进行 1000 次重采样
  - 输出指标均值、标准差、95% 分位数区间，并绘制直方图

## 特征与数据
- 目标列：自动识别二分类标签（默认为 Chronic_pain；若不存在则根据常见列名检测）
- 主要特征（与脚本中 SELECTED_FEATURES 同步，存在即用）：
  - IL6、IL10、TNFalpha、CRP、ACTH、PTC
  - 若干比值特征：IL6/IL10、TNFalpha/IL10、CRP/IL10、PTC/ACTH、PTC/IL6、PTC/CRP、IL6/TNFalpha、CRP/IL6、ACTH/IL6
- 预处理：数值缺失VAE + 标准化；类别/二元特征自动识别与 One-Hot

## 指标与可视化
- 指标：ROC AUC、PR AUC、Brier、Accuracy、Precision/PPV、Recall/Sensitivity、Specificity、F1、NPV、Youden
- 可视化：
  - 5 折 CV 的小提琴图（每模型、各指标）
  - 测试集 Bootstrap 的直方图与正态拟合曲线
- 阈值策略：
  - F1 最优（根据验证集或 OOF 动态搜索）
  - Youden 最优（Sensitivity + Specificity - 1）
  - Recall≥0.80 下 PPV 最大（面向高召回约束）

## 输出文件说明
- 基线结果目录：d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\baseline_model_comparison\
  - model_comparison_summary.csv（各模型测试表现 + CV 均值/标准差）
  - violin_*.png/.tiff（基线模型的小提琴图）
- 集成结果目录：d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\stacking_model_comparsion\
  - stacking_model_comparison_summary.csv（含各模型阈值 Thr_F1 与 Thr_Youden）
  - cv_results\cv_fold_results.csv、cv_fold_results_bounded.csv
  - violin_Ensemble__logit_convex+T__GB+RF__cv_f1.png（标题：HemoPain-Ensemble）
  - violin_Ensemble__logit_convex+T_AUC__GB+RF__cv_f1.png（标题：ENS-LogitT-AUC）
  - violin_Ensemble__rank_average+Platt__GB+RF__cv_f1.png（标题：ENS-RankPlatt）
  - violin_Ensemble__Stacking_LR__GB+RF__cv_f1.png（标题：ENS-Stack）
  - final_ensemble_convex_T.pkl / final_ensemble_rank_platt.pkl / final_ensemble_stacking_lr.pkl
- Bootstrap 输出目录：d:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\bootstrap_reslut\
  - test_metrics_convex_T.csv（单次测试点）
  - bootstrap_metrics_convex_T.csv（每次重采样）
  - bootstrap_summary_convex_T.csv（均值 / SD / 2.5%-97.5% 分位区间）
  - hist_roc_auc_convex_T.png、hist_pr_auc_convex_T.png、hist_f1_f1thr_convex_T.png 等直方图

## 复现实验流程建议
1. 确认 train.csv / test.csv 可用且目标列为 0/1
2. 运行 stacking_competition.py 生成模型与汇总，检查 cv_results 与 violin 图
3. 运行 competition.py 比较基线模型并生成对应可视化
4. 运行 bootstrap_rank_platt.py 生成测试集的 CI 与直方图


## 依赖与环境
- Python 3.10+（建议）
- 主要依赖：numpy、pandas、scikit-learn、scipy、seaborn、matplotlib、joblib
- 可选：catboost（若启用 CatBoost 相关功能）


## 参考与说明
- 实验设计与研究背景：详见 PDF  
  [2026.03.16.-YHarvest-疼痛论文.pdf]
- 关键脚本：  
  [competition.py](file:///d:/yiqy/sleepProjexts/发文章版_疼痛二分类任务/最终版本/competition.py)  
  [stacking_competition.py](file:///d:/yiqy/sleepProjexts/发文章版_疼痛二分类任务/最终版本/stacking_competition.py)  
  [bootstrap_rank_platt.py](file:///d:/yiqy/sleepProjexts/发文章版_疼痛二分类任务/最终版本/bootstrap_rank_platt.py)

