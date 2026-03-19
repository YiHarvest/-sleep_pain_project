#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据分析脚本
Advanced Data Analysis Script


版本: 1.0
创建时间: 2025-01-19

功能:
- 任务A: 比较插补策略对检验结论的影响
- 任务B: 检验变换对检验结论的影响  
- 任务C: ACTH效应量置信区间与p值稳定性分析
- 任务D: 单变量AUC的bootstrap置信区间分析
"""

import os
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 统计和机器学习库
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, ttest_ind
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedDataAnalyzer:
    """高级数据分析器"""
    
    def __init__(self, 
                 base_dir: str = "datasetprocess_results",
                 output_dir: str = "datasetprocess1_results",
                 random_state: int = 42,
                 alpha: float = 0.05):
        """
        初始化高级数据分析器
        
        参数:
        - base_dir: 基础数据目录
        - output_dir: 输出目录
        - random_state: 随机种子
        - alpha: 统计检验显著性水平
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.alpha = alpha
        
        # 生物指标列表
        self.biomarkers = ['IL6', 'IL10', 'TNFalpha', 'CRP', 'ACTH', 'PTC']
        
        # 创建输出目录
        self._create_directories()
        
        # 设置随机种子
        np.random.seed(self.random_state)
        
    def _create_directories(self):
        """创建必要的输出目录"""
        directories = [
            self.output_dir,
            self.output_dir / "compare_imputation",
            self.output_dir / "transform_compare", 
            self.output_dir / "bootstrap",
            self.output_dir / "plots"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """计算Cliff's delta效应量"""
        n1, n2 = len(group1), len(group2)
        dominance = 0
        
        for x in group1:
            for y in group2:
                if x > y:
                    dominance += 1
                elif x < y:
                    dominance -= 1
                    
        return dominance / (n1 * n2)
    
    def perform_statistical_test(self, group1: np.ndarray, group2: np.ndarray, 
                                feature_name: str) -> Dict[str, Any]:
        """执行完整的统计检验流程"""
        results = {
            'feature': feature_name,
            'n_group0': len(group1),
            'n_group1': len(group2),
            'mean_group0': np.mean(group1),
            'mean_group1': np.mean(group2),
            'std_group0': np.std(group1, ddof=1),
            'std_group1': np.std(group2, ddof=1)
        }
        
        # 正态性检验
        if len(group1) >= 3:
            shapiro_0_stat, shapiro_0_p = shapiro(group1)
        else:
            shapiro_0_stat, shapiro_0_p = np.nan, np.nan
            
        if len(group2) >= 3:
            shapiro_1_stat, shapiro_1_p = shapiro(group2)
        else:
            shapiro_1_stat, shapiro_1_p = np.nan, np.nan
            
        results.update({
            'shapiro_0_stat': shapiro_0_stat,
            'shapiro_0_p': shapiro_0_p,
            'shapiro_1_stat': shapiro_1_stat,
            'shapiro_1_p': shapiro_1_p
        })
        
        # 方差齐性检验
        try:
            levene_stat, levene_p = levene(group1, group2)
        except:
            levene_stat, levene_p = np.nan, np.nan
            
        results.update({
            'levene_stat': levene_stat,
            'levene_p': levene_p
        })
        
        # 判断是否使用参数检验
        normal_assumption = (shapiro_0_p > self.alpha and shapiro_1_p > self.alpha)
        equal_var_assumption = (levene_p > self.alpha)
        
        if normal_assumption and equal_var_assumption:
            # t检验
            test_stat, test_p = ttest_ind(group1, group2, equal_var=True)
            test_type = 't_test'
            effect_size = self.cohens_d(group1, group2)
            effect_type = 'cohens_d'
        else:
            # Mann-Whitney U检验
            test_stat, test_p = mannwhitneyu(group1, group2, alternative='two-sided')
            test_type = 'mann_whitney'
            effect_size = self.cliffs_delta(group1, group2)
            effect_type = 'cliffs_delta'
            
        results.update({
            'test_type': test_type,
            'test_stat': test_stat,
            'test_p': test_p,
            'effect_size': effect_size,
            'effect_type': effect_type
        })
        
        return results
    
    def task_a_compare_imputation(self):
        """任务A: 比较插补策略对检验结论的影响"""
        print("执行任务A: 比较插补策略对检验结论的影响...")
        
        # 读取两种插补数据
        median_file = self.base_dir / "cleaned_imputed_data_median.csv"
        mice_file = self.base_dir / "cleaned_imputed_data_mice.csv"
        
        data_median = pd.read_csv(median_file)
        data_mice = pd.read_csv(mice_file)
        
        print(f"中位数插补数据形状: {data_median.shape}")
        print(f"MICE插补数据形状: {data_mice.shape}")
        
        # 存储结果
        comparison_results = []
        
        for feature in self.biomarkers:
            if feature in data_median.columns and feature in data_mice.columns:
                print(f"分析特征: {feature}")
                
                # 中位数插补结果
                group0_median = data_median[data_median['Chronic_pain'] == 0][feature].dropna()
                group1_median = data_median[data_median['Chronic_pain'] == 1][feature].dropna()
                
                median_results = self.perform_statistical_test(
                    group0_median.values, group1_median.values, f"{feature}_median"
                )
                median_results['imputation_method'] = 'median'
                
                # MICE插补结果
                group0_mice = data_mice[data_mice['Chronic_pain'] == 0][feature].dropna()
                group1_mice = data_mice[data_mice['Chronic_pain'] == 1][feature].dropna()
                
                mice_results = self.perform_statistical_test(
                    group0_mice.values, group1_mice.values, f"{feature}_mice"
                )
                mice_results['imputation_method'] = 'mice'
                
                comparison_results.extend([median_results, mice_results])
        
        # 转换为DataFrame
        results_df = pd.DataFrame(comparison_results)
        
        # FDR校正
        median_ps = results_df[results_df['imputation_method'] == 'median']['test_p'].values
        mice_ps = results_df[results_df['imputation_method'] == 'mice']['test_p'].values
        
        # 分别进行FDR校正
        if len(median_ps) > 0:
            _, median_fdr_ps, _, _ = multipletests(median_ps, alpha=self.alpha, method='fdr_bh')
            results_df.loc[results_df['imputation_method'] == 'median', 'fdr_p'] = median_fdr_ps
            
        if len(mice_ps) > 0:
            _, mice_fdr_ps, _, _ = multipletests(mice_ps, alpha=self.alpha, method='fdr_bh')
            results_df.loc[results_df['imputation_method'] == 'mice', 'fdr_p'] = mice_fdr_ps
        
        # 添加显著性标记
        results_df['significant_raw'] = results_df['test_p'] < self.alpha
        results_df['significant_fdr'] = results_df['fdr_p'] < self.alpha
        
        # 保存结果
        output_file = self.output_dir / "compare_imputation" / "04_univariate_tests_median_vs_mice.csv"
        results_df.to_csv(output_file, index=False)
        
        # 生成总结报告
        self._generate_imputation_summary(results_df)
        
        print(f"任务A完成，结果保存到: {output_file}")
        return results_df
    
    def _generate_imputation_summary(self, results_df: pd.DataFrame):
        """生成插补比较总结报告"""
        summary_file = self.output_dir / "compare_imputation" / "summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 插补策略比较总结报告 ===\n\n")
            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"显著性水平: {self.alpha}\n\n")
            
            # 按特征分组比较
            features = [col.replace('_median', '').replace('_mice', '') 
                       for col in results_df['feature'].unique()]
            features = list(set(features))
            
            consistent_features = []
            inconsistent_features = []
            
            for feature in features:
                median_row = results_df[results_df['feature'] == f"{feature}_median"]
                mice_row = results_df[results_df['feature'] == f"{feature}_mice"]
                
                if len(median_row) > 0 and len(mice_row) > 0:
                    median_sig = median_row['significant_fdr'].iloc[0]
                    mice_sig = mice_row['significant_fdr'].iloc[0]
                    
                    median_p = median_row['fdr_p'].iloc[0]
                    mice_p = mice_row['fdr_p'].iloc[0]
                    
                    median_effect = median_row['effect_size'].iloc[0]
                    mice_effect = mice_row['effect_size'].iloc[0]
                    
                    if median_sig == mice_sig:
                        consistent_features.append(feature)
                        f.write(f"{feature}: 结果一致 ({'显著' if median_sig else '不显著'})\n")
                        f.write(f"  中位数插补: p={median_p:.4f}, 效应量={median_effect:.4f}\n")
                        f.write(f"  MICE插补: p={mice_p:.4f}, 效应量={mice_effect:.4f}\n\n")
                    else:
                        inconsistent_features.append(feature)
                        f.write(f"{feature}: 结果不一致\n")
                        f.write(f"  中位数插补: p={median_p:.4f} ({'显著' if median_sig else '不显著'}), 效应量={median_effect:.4f}\n")
                        f.write(f"  MICE插补: p={mice_p:.4f} ({'显著' if mice_sig else '不显著'}), 效应量={mice_effect:.4f}\n")
                        f.write(f"  建议: 需要谨慎解释结果，考虑插补方法的影响\n\n")
            
            f.write(f"\n总结:\n")
            f.write(f"- 结果一致的特征 ({len(consistent_features)}个): {', '.join(consistent_features)}\n")
            f.write(f"- 结果不一致的特征 ({len(inconsistent_features)}个): {', '.join(inconsistent_features)}\n")
            
            if len(inconsistent_features) > 0:
                f.write(f"\n重要提醒: {len(inconsistent_features)}个特征的显著性结论依赖于插补策略，")
                f.write(f"在后续分析中需要报告敏感性分析结果。\n")
    
    def task_b_transform_compare(self):
        """任务B: 检验变换对检验结论的影响"""
        print("执行任务B: 检验变换对检验结论的影响...")
        
        # 读取MICE插补数据作为基础
        mice_file = self.base_dir / "cleaned_imputed_data_mice.csv"
        data = pd.read_csv(mice_file)
        
        print(f"数据形状: {data.shape}")
        
        # 存储结果
        transform_results = []
        
        for feature in self.biomarkers:
            if feature in data.columns:
                print(f"分析特征: {feature}")
                
                # 原始数据
                group0_raw = data[data['Chronic_pain'] == 0][feature].dropna()
                group1_raw = data[data['Chronic_pain'] == 1][feature].dropna()
                
                raw_results = self.perform_statistical_test(
                    group0_raw.values, group1_raw.values, f"{feature}_raw"
                )
                raw_results['transform'] = 'raw'
                
                # log1p变换数据
                group0_log = np.log1p(group0_raw)
                group1_log = np.log1p(group1_raw)
                
                log_results = self.perform_statistical_test(
                    group0_log.values, group1_log.values, f"{feature}_log1p"
                )
                log_results['transform'] = 'log1p'
                
                transform_results.extend([raw_results, log_results])
        
        # 转换为DataFrame
        results_df = pd.DataFrame(transform_results)
        
        # FDR校正
        raw_ps = results_df[results_df['transform'] == 'raw']['test_p'].values
        log_ps = results_df[results_df['transform'] == 'log1p']['test_p'].values
        
        if len(raw_ps) > 0:
            _, raw_fdr_ps, _, _ = multipletests(raw_ps, alpha=self.alpha, method='fdr_bh')
            results_df.loc[results_df['transform'] == 'raw', 'fdr_p'] = raw_fdr_ps
            
        if len(log_ps) > 0:
            _, log_fdr_ps, _, _ = multipletests(log_ps, alpha=self.alpha, method='fdr_bh')
            results_df.loc[results_df['transform'] == 'log1p', 'fdr_p'] = log_fdr_ps
        
        # 添加显著性标记
        results_df['significant_raw'] = results_df['test_p'] < self.alpha
        results_df['significant_fdr'] = results_df['fdr_p'] < self.alpha
        
        # 保存结果
        output_file = self.output_dir / "transform_compare" / "raw_vs_log_tests.csv"
        results_df.to_csv(output_file, index=False)
        
        # 生成总结报告
        self._generate_transform_summary(results_df)
        
        print(f"任务B完成，结果保存到: {output_file}")
        return results_df
    
    def _generate_transform_summary(self, results_df: pd.DataFrame):
        """生成变换比较总结报告"""
        summary_file = self.output_dir / "transform_compare" / "summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 数据变换影响分析总结报告 ===\n\n")
            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"显著性水平: {self.alpha}\n\n")
            
            # 按特征分组比较
            features = [col.replace('_raw', '').replace('_log1p', '') 
                       for col in results_df['feature'].unique()]
            features = list(set(features))
            
            improved_by_transform = []
            worsened_by_transform = []
            no_change = []
            
            for feature in features:
                raw_row = results_df[results_df['feature'] == f"{feature}_raw"]
                log_row = results_df[results_df['feature'] == f"{feature}_log1p"]
                
                if len(raw_row) > 0 and len(log_row) > 0:
                    raw_sig = raw_row['significant_fdr'].iloc[0]
                    log_sig = log_row['significant_fdr'].iloc[0]
                    
                    raw_p = raw_row['fdr_p'].iloc[0]
                    log_p = log_row['fdr_p'].iloc[0]
                    
                    raw_effect = abs(raw_row['effect_size'].iloc[0])
                    log_effect = abs(log_row['effect_size'].iloc[0])
                    
                    f.write(f"{feature}:\n")
                    f.write(f"  原始数据: p={raw_p:.4f} ({'显著' if raw_sig else '不显著'}), |效应量|={raw_effect:.4f}\n")
                    f.write(f"  log1p变换: p={log_p:.4f} ({'显著' if log_sig else '不显著'}), |效应量|={log_effect:.4f}\n")
                    
                    # 判断变换效果
                    if not raw_sig and log_sig:
                        improved_by_transform.append(feature)
                        f.write(f"  结论: log1p变换使特征变为显著\n")
                    elif raw_sig and not log_sig:
                        worsened_by_transform.append(feature)
                        f.write(f"  结论: log1p变换使特征失去显著性\n")
                    elif raw_sig == log_sig:
                        no_change.append(feature)
                        if log_effect > raw_effect * 1.1:
                            f.write(f"  结论: 显著性不变，但效应量增强 ({raw_effect:.3f} → {log_effect:.3f})\n")
                        elif log_effect < raw_effect * 0.9:
                            f.write(f"  结论: 显著性不变，但效应量减弱 ({raw_effect:.3f} → {log_effect:.3f})\n")
                        else:
                            f.write(f"  结论: 变换对结果影响较小\n")
                    f.write("\n")
            
            f.write(f"总结:\n")
            f.write(f"- 因log1p变换改善的特征 ({len(improved_by_transform)}个): {', '.join(improved_by_transform)}\n")
            f.write(f"- 因log1p变换恶化的特征 ({len(worsened_by_transform)}个): {', '.join(worsened_by_transform)}\n")
            f.write(f"- 变换影响较小的特征 ({len(no_change)}个): {', '.join(no_change)}\n")
            
            if len(improved_by_transform) > 0:
                f.write(f"\n建议: 对于{', '.join(improved_by_transform)}，")
                f.write(f"在后续建模中考虑使用log1p变换。\n")
    
    def task_c_acth_bootstrap(self):
        """任务C: ACTH效应量置信区间与p值稳定性分析"""
        print("执行任务C: ACTH效应量置信区间与p值稳定性分析...")
        
        # 读取MICE插补数据
        mice_file = self.base_dir / "cleaned_imputed_data_mice.csv"
        data = pd.read_csv(mice_file)
        
        # 提取ACTH数据
        acth_data = data[['Chronic_pain', 'ACTH']].dropna()
        group0 = acth_data[acth_data['Chronic_pain'] == 0]['ACTH'].values
        group1 = acth_data[acth_data['Chronic_pain'] == 1]['ACTH'].values
        
        print(f"ACTH数据: 非疼痛组 n={len(group0)}, 疼痛组 n={len(group1)}")
        
        # Bootstrap参数
        n_bootstrap = 2000
        bootstrap_results = []
        
        print(f"执行{n_bootstrap}次bootstrap重采样...")
        
        for i in range(n_bootstrap):
            if (i + 1) % 500 == 0:
                print(f"  完成 {i + 1}/{n_bootstrap} 次重采样")
                
            # 重采样
            boot_group0 = resample(group0, n_samples=len(group0), random_state=i)
            boot_group1 = resample(group1, n_samples=len(group1), random_state=i+n_bootstrap)
            
            # 计算效应量
            mean_diff = np.mean(boot_group1) - np.mean(boot_group0)
            cliffs_d = self.cliffs_delta(boot_group0, boot_group1)
            
            # 执行统计检验
            try:
                _, p_value = mannwhitneyu(boot_group0, boot_group1, alternative='two-sided')
            except:
                p_value = np.nan
                
            bootstrap_results.append({
                'iteration': i,
                'mean_diff': mean_diff,
                'cliffs_delta': cliffs_d,
                'p_value': p_value,
                'significant': p_value < self.alpha if not np.isnan(p_value) else False
            })
        
        # 转换为DataFrame
        boot_df = pd.DataFrame(bootstrap_results)
        
        # 计算置信区间
        mean_diff_ci = np.percentile(boot_df['mean_diff'].dropna(), [2.5, 97.5])
        cliffs_ci = np.percentile(boot_df['cliffs_delta'].dropna(), [2.5, 97.5])
        
        # 计算统计量
        significance_rate = boot_df['significant'].mean()
        
        # 保存bootstrap结果
        boot_file = self.output_dir / "bootstrap" / "ACTH_boot_ci.csv"
        
        # 创建汇总统计
        summary_stats = {
            'metric': ['mean_difference', 'cliffs_delta', 'p_value', 'significance_rate'],
            'mean': [boot_df['mean_diff'].mean(), boot_df['cliffs_delta'].mean(), 
                    boot_df['p_value'].mean(), significance_rate],
            'std': [boot_df['mean_diff'].std(), boot_df['cliffs_delta'].std(), 
                   boot_df['p_value'].std(), np.nan],
            'ci_lower': [mean_diff_ci[0], cliffs_ci[0], 
                        np.percentile(boot_df['p_value'].dropna(), 2.5), np.nan],
            'ci_upper': [mean_diff_ci[1], cliffs_ci[1], 
                        np.percentile(boot_df['p_value'].dropna(), 97.5), np.nan]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(boot_file, index=False)
        
        # 绘制分布图
        self._plot_acth_bootstrap(boot_df, mean_diff_ci, cliffs_ci)
        
        # 生成报告
        self._generate_acth_bootstrap_report(boot_df, summary_df, significance_rate)
        
        print(f"任务C完成，结果保存到: {boot_file}")
        return boot_df, summary_df
    
    def _plot_acth_bootstrap(self, boot_df: pd.DataFrame, 
                            mean_diff_ci: Tuple[float, float], 
                            cliffs_ci: Tuple[float, float]):
        """绘制ACTH bootstrap分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 均值差分布
        axes[0, 0].hist(boot_df['mean_diff'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(mean_diff_ci[0], color='red', linestyle='--', label=f'95% CI: [{mean_diff_ci[0]:.3f}, {mean_diff_ci[1]:.3f}]')
        axes[0, 0].axvline(mean_diff_ci[1], color='red', linestyle='--')
        axes[0, 0].axvline(0, color='black', linestyle='-', alpha=0.5, label='无差异')
        axes[0, 0].set_xlabel('均值差 (疼痛组 - 非疼痛组)')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].set_title('ACTH均值差Bootstrap分布')
        axes[0, 0].legend()
        
        # Cliff's delta分布
        axes[0, 1].hist(boot_df['cliffs_delta'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(cliffs_ci[0], color='red', linestyle='--', label=f'95% CI: [{cliffs_ci[0]:.3f}, {cliffs_ci[1]:.3f}]')
        axes[0, 1].axvline(cliffs_ci[1], color='red', linestyle='--')
        axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.5, label='无效应')
        axes[0, 1].set_xlabel("Cliff's Delta")
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title("ACTH Cliff's Delta Bootstrap分布")
        axes[0, 1].legend()
        
        # p值分布
        axes[1, 0].hist(boot_df['p_value'].dropna(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].axvline(self.alpha, color='red', linestyle='--', label=f'α = {self.alpha}')
        axes[1, 0].set_xlabel('p值')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('ACTH p值Bootstrap分布')
        axes[1, 0].legend()
        
        # 显著性比例
        sig_counts = boot_df['significant'].value_counts()
        labels = ['不显著', '显著']
        colors = ['lightcoral', 'lightgreen']
        axes[1, 1].pie([sig_counts.get(False, 0), sig_counts.get(True, 0)], 
                      labels=labels, colors=colors, autopct='%1.1f%%')
        axes[1, 1].set_title('Bootstrap显著性比例')
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.output_dir / "bootstrap" / "ACTH_boot_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ACTH bootstrap分布图保存到: {plot_file}")
    
    def _generate_acth_bootstrap_report(self, boot_df: pd.DataFrame, 
                                      summary_df: pd.DataFrame, 
                                      significance_rate: float):
        """生成ACTH bootstrap分析报告"""
        report_file = self.output_dir / "bootstrap" / "ACTH_bootstrap_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== ACTH效应量置信区间与p值稳定性分析报告 ===\n\n")
            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bootstrap重采样次数: {len(boot_df)}\n")
            f.write(f"显著性水平: {self.alpha}\n\n")
            
            # 效应量分析
            mean_diff_row = summary_df[summary_df['metric'] == 'mean_difference'].iloc[0]
            cliffs_row = summary_df[summary_df['metric'] == 'cliffs_delta'].iloc[0]
            
            f.write("效应量分析:\n")
            f.write(f"1. 均值差 (疼痛组 - 非疼痛组):\n")
            f.write(f"   均值: {mean_diff_row['mean']:.4f}\n")
            f.write(f"   95% CI: [{mean_diff_row['ci_lower']:.4f}, {mean_diff_row['ci_upper']:.4f}]\n")
            
            if mean_diff_row['ci_lower'] > 0:
                f.write(f"   结论: 疼痛组ACTH显著高于非疼痛组 (CI不包含0)\n")
            elif mean_diff_row['ci_upper'] < 0:
                f.write(f"   结论: 疼痛组ACTH显著低于非疼痛组 (CI不包含0)\n")
            else:
                f.write(f"   结论: 两组ACTH差异不稳定 (CI包含0)\n")
            
            f.write(f"\n2. Cliff's Delta (非参数效应量):\n")
            f.write(f"   均值: {cliffs_row['mean']:.4f}\n")
            f.write(f"   95% CI: [{cliffs_row['ci_lower']:.4f}, {cliffs_row['ci_upper']:.4f}]\n")
            
            cliffs_mean = cliffs_row['mean']
            if abs(cliffs_mean) < 0.147:
                effect_size_interp = "可忽略"
            elif abs(cliffs_mean) < 0.33:
                effect_size_interp = "小"
            elif abs(cliffs_mean) < 0.474:
                effect_size_interp = "中等"
            else:
                effect_size_interp = "大"
                
            f.write(f"   效应量大小: {effect_size_interp}\n")
            
            # 显著性稳定性分析
            f.write(f"\n显著性稳定性分析:\n")
            f.write(f"显著性比例: {significance_rate:.1%} ({int(significance_rate * len(boot_df))}/{len(boot_df)})\n")
            
            if significance_rate >= 0.95:
                stability = "非常稳定"
            elif significance_rate >= 0.80:
                stability = "稳定"
            elif significance_rate >= 0.60:
                stability = "中等稳定"
            else:
                stability = "不稳定"
                
            f.write(f"稳定性评价: {stability}\n")
            
            # 总结建议
            f.write(f"\n总结与建议:\n")
            if significance_rate >= 0.80 and cliffs_row['ci_lower'] > 0:
                f.write(f"ACTH在疼痛预测中显示出稳定的显著性和正向效应。\n")
                f.write(f"建议在后续建模中重点考虑ACTH作为预测特征。\n")
            elif significance_rate >= 0.60:
                f.write(f"ACTH显示出中等程度的稳定性，但需要谨慎解释。\n")
                f.write(f"建议结合其他特征进行综合分析。\n")
            else:
                f.write(f"ACTH的显著性不够稳定，可能存在过拟合风险。\n")
                f.write(f"建议增加样本量或考虑其他特征。\n")
    
    def task_d_auc_bootstrap(self):
        """任务D: 单变量AUC的bootstrap置信区间分析"""
        print("执行任务D: 单变量AUC的bootstrap置信区间分析...")
        
        # 读取MICE插补数据
        mice_file = self.base_dir / "cleaned_imputed_data_mice.csv"
        data = pd.read_csv(mice_file)
        
        # Bootstrap参数
        n_bootstrap = 1000
        auc_results = []
        
        for feature in self.biomarkers:
            if feature in data.columns:
                print(f"分析特征: {feature}")
                
                # 准备数据
                feature_data = data[['Chronic_pain', feature]].dropna()
                y_true = feature_data['Chronic_pain'].values
                y_scores = feature_data[feature].values
                
                # 原始AUC
                try:
                    original_auc = roc_auc_score(y_true, y_scores)
                except:
                    original_auc = np.nan
                
                # Bootstrap AUC
                bootstrap_aucs = []
                bootstrap_rocs = []
                
                for i in range(n_bootstrap):
                    # 重采样
                    indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)
                    boot_y_true = y_true[indices]
                    boot_y_scores = y_scores[indices]
                    
                    try:
                        # 计算AUC
                        boot_auc = roc_auc_score(boot_y_true, boot_y_scores)
                        bootstrap_aucs.append(boot_auc)
                        
                        # 计算ROC曲线 (每10次保存一次，避免内存过大)
                        if i % 10 == 0:
                            fpr, tpr, _ = roc_curve(boot_y_true, boot_y_scores)
                            bootstrap_rocs.append((fpr, tpr))
                    except:
                        continue
                
                # 计算置信区间
                if len(bootstrap_aucs) > 0:
                    auc_mean = np.mean(bootstrap_aucs)
                    auc_std = np.std(bootstrap_aucs)
                    auc_ci = np.percentile(bootstrap_aucs, [2.5, 97.5])
                    
                    auc_results.append({
                        'feature': feature,
                        'original_auc': original_auc,
                        'bootstrap_mean_auc': auc_mean,
                        'bootstrap_std_auc': auc_std,
                        'ci_lower': auc_ci[0],
                        'ci_upper': auc_ci[1],
                        'n_bootstrap': len(bootstrap_aucs)
                    })
                    
                    # 绘制ROC曲线
                    self._plot_roc_with_ci(feature, y_true, y_scores, bootstrap_rocs, 
                                         original_auc, auc_ci)
        
        # 保存结果
        results_df = pd.DataFrame(auc_results)
        output_file = self.output_dir / "08_univariate_auc_ci.csv"
        results_df.to_csv(output_file, index=False)
        
        # 生成总结报告
        self._generate_auc_summary(results_df)
        
        print(f"任务D完成，结果保存到: {output_file}")
        return results_df
    
    def _plot_roc_with_ci(self, feature: str, y_true: np.ndarray, y_scores: np.ndarray,
                         bootstrap_rocs: List[Tuple], original_auc: float, auc_ci: Tuple[float, float]):
        """绘制带置信区间的ROC曲线"""
        plt.figure(figsize=(8, 8))
        
        # 原始ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, color='blue', linewidth=2, 
                label=f'原始ROC (AUC = {original_auc:.3f})')
        
        # Bootstrap ROC曲线 (绘制部分)
        if len(bootstrap_rocs) > 0:
            for i, (boot_fpr, boot_tpr) in enumerate(bootstrap_rocs[:20]):  # 只绘制前20条
                plt.plot(boot_fpr, boot_tpr, color='gray', alpha=0.1, linewidth=0.5)
        
        # 对角线
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5, label='随机分类器')
        
        # 设置图形
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (1-特异性)')
        plt.ylabel('真阳性率 (敏感性)')
        plt.title(f'{feature} - ROC曲线\nAUC 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plot_file = self.output_dir / "plots" / f"roc_{feature}_ci.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ROC曲线保存到: {plot_file}")
    
    def _generate_auc_summary(self, results_df: pd.DataFrame):
        """生成AUC分析总结报告"""
        summary_file = self.output_dir / "auc_bootstrap_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== 单变量AUC Bootstrap置信区间分析报告 ===\n\n")
            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bootstrap重采样次数: 1000\n\n")
            
            # 按AUC排序
            results_sorted = results_df.sort_values('bootstrap_mean_auc', ascending=False)
            
            f.write("各特征AUC分析结果:\n")
            f.write("-" * 60 + "\n")
            
            for _, row in results_sorted.iterrows():
                feature = row['feature']
                original_auc = row['original_auc']
                mean_auc = row['bootstrap_mean_auc']
                ci_lower = row['ci_lower']
                ci_upper = row['ci_upper']
                
                f.write(f"{feature}:\n")
                f.write(f"  原始AUC: {original_auc:.4f}\n")
                f.write(f"  Bootstrap均值AUC: {mean_auc:.4f}\n")
                f.write(f"  95% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
                
                # 判断预测能力
                if ci_lower > 0.7:
                    performance = "良好"
                elif ci_lower > 0.6:
                    performance = "中等"
                elif ci_lower > 0.5:
                    performance = "较差"
                else:
                    performance = "无预测能力"
                    
                f.write(f"  预测能力评价: {performance}\n")
                
                # 稳定性评价
                ci_width = ci_upper - ci_lower
                if ci_width < 0.1:
                    stability = "稳定"
                elif ci_width < 0.2:
                    stability = "中等稳定"
                else:
                    stability = "不稳定"
                    
                f.write(f"  稳定性评价: {stability} (CI宽度: {ci_width:.4f})\n\n")
            
            # 总结排名
            f.write("预测能力排名:\n")
            for i, (_, row) in enumerate(results_sorted.iterrows(), 1):
                f.write(f"{i}. {row['feature']}: AUC = {row['bootstrap_mean_auc']:.4f} "
                       f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]\n")
            
            # 建议
            best_features = results_sorted[results_sorted['ci_lower'] > 0.6]['feature'].tolist()
            if len(best_features) > 0:
                f.write(f"\n建议优先考虑的特征: {', '.join(best_features)}\n")
            else:
                f.write(f"\n注意: 所有单变量特征的预测能力都较为有限，建议考虑多变量建模。\n")
    
    def run_all_tasks(self):
        """运行所有分析任务"""
        print("开始执行高级数据分析...")
        print("=" * 60)
        
        try:
            # 任务A: 比较插补策略
            task_a_results = self.task_a_compare_imputation()
            
            print("\n" + "=" * 60)
            
            # 任务B: 比较变换影响
            task_b_results = self.task_b_transform_compare()
            
            print("\n" + "=" * 60)
            
            # 任务C: ACTH bootstrap分析
            task_c_results = self.task_c_acth_bootstrap()
            
            print("\n" + "=" * 60)
            
            # 任务D: AUC bootstrap分析
            task_d_results = self.task_d_auc_bootstrap()
            
            print("\n" + "=" * 60)
            print("所有分析任务完成!")
            
            return {
                'task_a': task_a_results,
                'task_b': task_b_results, 
                'task_c': task_c_results,
                'task_d': task_d_results
            }
            
        except Exception as e:
            print(f"分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("高级数据分析脚本")
    print("作者: AI Assistant")
    print("版本: 1.0")
    print("=" * 60)
    
    # 创建分析器
    analyzer = AdvancedDataAnalyzer(
        base_dir="datasetprocess_results",
        output_dir="datasetprocess1_results",
        random_state=42,
        alpha=0.05
    )
    
    # 运行所有任务
    results = analyzer.run_all_tasks()
    
    if results is not None:
        print("\n分析完成! 请查看 datasetprocess1_results 目录下的结果文件。")
    else:
        print("\n分析失败，请检查错误信息。")

if __name__ == "__main__":
    main()