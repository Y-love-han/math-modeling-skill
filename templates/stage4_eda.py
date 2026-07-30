# -*- coding: utf-8 -*-
"""
stage4_eda.py — 深度探索性数据分析（EDA）四层管线
用途：阶段 4 数据驱动洞察——单变量分布分析 + 双变量关系 + 多变量结构 + 网络特性
输入：preprocessed_data.csv（data_preprocessing.py 的产出）
输出：data_quality_report.md + >=3 张出版级 EDA 图表（>=300dpi，色盲友好配色）
运行方式：python stage4_eda.py --input 结果/preprocessed_data.csv
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')
from utils import (save_fig, COLORS, RNG, FIGURES_DIR, OUTPUT_DIR,
                   confidence_interval, bootstrap_ci)

N_BOOTSTRAP = 1000
ALPHA = 0.05
DISTRIBUTIONS = {
    'norm': 'Normal', 'lognorm': 'Log-normal', 'expon': 'Exponential',
    'gamma': 'Gamma', 'beta': 'Beta', 'weibull_min': 'Weibull',
}

def fit_distributions(data):
    """Fit multiple distributions and select the best using AIC/BIC"""
    results = []
    for dist_name, dist_label in DISTRIBUTIONS.items():
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            log_lik = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * log_lik
            bic = k * np.log(len(data)) - 2 * log_lik
            ks_stat, ks_p = stats.kstest(data, dist_name, args=params)
            results.append({
                'distribution': dist_label, 'aic': float(aic),
                'bic': float(bic), 'ks_pvalue': float(ks_p)
            })
        except Exception:
            pass
    return sorted(results, key=lambda x: x['aic'])

def plot_univariate_numeric(df, col, dist_results):
    """Plot histogram + KDE + QQ-plot + best-fit distribution for a numeric column"""
    data = df[col].dropna().values
    if len(data) < 5:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Histogram + KDE + best fit
    ax = axes[0]
    ax.hist(data, bins=min(50, int(np.sqrt(len(data)))), density=True,
            alpha=0.6, color=COLORS[0], edgecolor='white', label='Data')
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range), color=COLORS[1], linewidth=2, label='KDE')
    if dist_results:
        best = dist_results[0]
        try:
            dist = getattr(stats, best['distribution'].split()[0].lower()
                          if ' ' not in best['distribution'] else 'norm')
            params = dist.fit(data)
            ax.plot(x_range, dist.pdf(x_range, *params),
                    color=COLORS[2], linewidth=2, linestyle='--',
                    label=f"Best: {best['distribution']} (AIC={best['aic']:.1f})")
        except Exception:
            pass
    ax.set_xlabel(col); ax.set_ylabel('Density')
    ax.set_title(f'(a) Distribution of {col}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) QQ-plot vs Normal
    ax = axes[1]
    stats.probplot(data, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(COLORS[0])
    ax.get_lines()[0].set_markeredgecolor(COLORS[0])
    ax.get_lines()[1].set_color(COLORS[3])
    ax.set_title(f'(b) Q-Q Plot (Normal)'); ax.grid(True, alpha=0.3)

    # (c) Box plot + violin
    ax = axes[2]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.3)
    bp['boxes'][0].set_facecolor(COLORS[0])
    ax.set_title(f'(c) Box Plot: {col}')
    ax.set_ylabel(col); ax.grid(True, alpha=0.3)
    stats_text = f"n={len(data)}\nmean={np.mean(data):.3g}\nstd={np.std(data):.3g}\nskew={stats.skew(data):.3g}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

def univariate_analysis(df, max_cols=20):
    """Layer 1: Univariate distribution analysis"""
    results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    for col in numeric_cols:
        data = df[col].dropna().values
        if len(data) < 5:
            continue
        dist_results = fit_distributions(data)
        results[col] = {
            'n': len(data), 'missing': int(df[col].isna().sum()),
            'mean': float(np.mean(data)), 'std': float(np.std(data)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data)),
            'best_fit': dist_results[0] if dist_results else None
        }
        fig = plot_univariate_numeric(df, col, dist_results)
        if fig:
            save_fig(fig, f'figD1_{col}_dist.png')
    return results

def bivariate_analysis(df, max_cols=15):
    """Layer 2: Bivariate relationship analysis with 3 correlation methods"""
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    if len(numeric_cols) < 2:
        return results

    # Correlation matrix with 3 methods
    pearson = df[numeric_cols].corr(method='pearson')
    spearman = df[numeric_cols].corr(method='spearman')
    kendall = df[numeric_cols].corr(method='kendall')

    # Detect nonlinear signals: |Spearman - Pearson| > 0.15
    nonlinear_signals = []
    for i, c1 in enumerate(numeric_cols):
        for j, c2 in enumerate(numeric_cols):
            if i < j:
                diff = abs(spearman.loc[c1, c2] - pearson.loc[c1, c2])
                if diff > 0.15 and abs(spearman.loc[c1, c2]) > 0.3:
                    nonlinear_signals.append({
                        'pair': f'{c1} vs {c2}',
                        'pearson': float(pearson.loc[c1, c2]),
                        'spearman': float(spearman.loc[c1, c2]),
                        'kendall': float(kendall.loc[c1, c2]),
                        'delta': float(diff),
                        'interpretation': 'Nonlinear relationship detected'
                    })

    # Plot correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(numeric_cols) * 0.4)))
    import seaborn as sns
    mask = np.triu(np.ones_like(pearson, dtype=bool))
    sns.heatmap(pearson, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=axes[0],
                cbar_kws={'shrink': 0.8})
    axes[0].set_title('(a) Pearson Correlation')
    sns.heatmap(spearman, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, ax=axes[1],
                cbar_kws={'shrink': 0.8})
    axes[1].set_title('(b) Spearman Correlation')
    plt.tight_layout()
    save_fig(fig, 'figD2_correlation_matrix.png')

    # Scatter matrix for top-k correlated pairs
    top_pairs = []
    for i, c1 in enumerate(numeric_cols):
        for j, c2 in enumerate(numeric_cols):
            if i < j and abs(pearson.loc[c1, c2]) > 0.5:
                top_pairs.append((c1, c2, pearson.loc[c1, c2]))
    top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = top_pairs[:4]

    if top_pairs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for idx, (c1, c2, r) in enumerate(top_pairs):
            ax = axes[idx // 2][idx % 2]
            ax.scatter(df[c1], df[c2], alpha=0.5, s=10, color=COLORS[0])
            # Add regression line
            mask = df[c1].notna() & df[c2].notna()
            if mask.sum() > 2:
                from numpy.polynomial.polynomial import polyfit
                x_clean = df.loc[mask, c1].values
                y_clean = df.loc[mask, c2].values
                try:
                    b, m = polyfit(x_clean, y_clean, 1)
                    x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                    ax.plot(x_range, b + m * x_range, color=COLORS[3],
                            linewidth=2, linestyle='--')
                except Exception:
                    pass
            ax.set_xlabel(c1); ax.set_ylabel(c2)
            ax.set_title(f'r = {r:.3f}'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, 'figD2_scatter_pairs.png')

    return {
        'nonlinear_signals': nonlinear_signals,
        'n_pairs_strong_corr': len(top_pairs)
    }

def multivariate_analysis(df, max_cols=15):
    """Layer 3: PCA + t-SNE + clustering tendency"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    numeric_cols = [c for c in numeric_cols if df[c].isna().sum() / len(df) < 0.5]
    if len(numeric_cols) < 3:
        return None

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    data_scaled = StandardScaler().fit_transform(
        df[numeric_cols].fillna(df[numeric_cols].median()))

    # PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Scree plot
    ax = axes[0]
    explained = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained)
    ax.bar(range(1, len(explained) + 1), explained, color=COLORS[0], alpha=0.7, label='Individual')
    ax.plot(range(1, len(explained) + 1), cumsum, 'o-', color=COLORS[3], linewidth=2, label='Cumulative')
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_xlabel('Principal Component'); ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('(a) PCA Scree Plot'); ax.legend(); ax.grid(True, alpha=0.3)

    # (b) PCA biplot (PC1 vs PC2)
    ax = axes[1]
    ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=15, color=COLORS[0])
    ax.set_xlabel(f'PC1 ({explained[0]:.1%})')
    ax.set_ylabel(f'PC2 ({explained[1]:.1%})')
    ax.set_title('(b) PCA: PC1 vs PC2'); ax.grid(True, alpha=0.3)
    # Add loading vectors for top features
    loadings = pca.components_[:2].T
    top_idx = np.argsort(np.abs(loadings[:, 0]) + np.abs(loadings[:, 1]))[-5:]
    for i in top_idx:
        ax.arrow(0, 0, loadings[i, 0] * pca_result[:, 0].std() * 3,
                 loadings[i, 1] * pca_result[:, 1].std() * 3,
                 color=COLORS[3], alpha=0.7, width=0.02, head_width=0.1)
        ax.text(loadings[i, 0] * pca_result[:, 0].std() * 3.5,
                loadings[i, 1] * pca_result[:, 1].std() * 3.5,
                numeric_cols[i][:15], fontsize=7, color=COLORS[3])

    # (c) t-SNE (if sklearn available)
    ax = axes[2]
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_scaled) // 3))
        tsne_result = tsne.fit_transform(data_scaled)
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5, s=15, color=COLORS[1])
        ax.set_title('(c) t-SNE Visualization')
    except Exception:
        ax.text(0.5, 0.5, 't-SNE not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(c) t-SNE (skipped)')
    ax.set_xlabel('Dimension 1'); ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'figD3_multivariate.png')

    return {
        'explained_variance_ratio': [float(x) for x in explained[:5]],
        'n_components_80pct': int(np.searchsorted(cumsum, 0.8) + 1),
        'top_pc1_features': [numeric_cols[i] for i in np.argsort(np.abs(pca.components_[0]))[-5:]],
    }

def network_analysis(df):
    """Layer 4: Network characteristics (if applicable)"""
    # Attempt to detect network/graph structure in data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 4:
        return None

    # Use correlation as adjacency
    corr = df[numeric_cols].corr().abs()
    # Threshold to create sparse adjacency
    threshold = np.percentile(corr.values[corr.values < 1], 90)
    adj = (corr > threshold).astype(int)
    np.fill_diagonal(adj.values, 0)

    try:
        import networkx as nx
        G = nx.from_pandas_adjacency(adj)
        metrics = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'n_components': nx.number_connected_components(G),
        }

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # (a) Network graph
        ax = axes[0]
        pos = nx.spring_layout(G, seed=42, k=1.5)
        degrees = dict(G.degree())
        node_sizes = [v * 300 + 100 for v in degrees.values()]
        nx.draw(G, pos, ax=ax, node_color=COLORS[0], node_size=node_sizes,
                edge_color='gray', alpha=0.7, with_labels=True,
                font_size=8, font_color='black')
        ax.set_title(f'(a) Correlation Network (>{threshold:.2f})')

        # (b) Degree distribution
        ax = axes[1]
        deg_values = list(dict(G.degree()).values())
        ax.hist(deg_values, bins=min(20, len(set(deg_values))),
                color=COLORS[0], alpha=0.7, edgecolor='white')
        ax.set_xlabel('Degree'); ax.set_ylabel('Frequency')
        ax.set_title('(b) Degree Distribution'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_fig(fig, 'figD4_network.png')
        return metrics
    except ImportError:
        return {'warning': 'networkx not available'}

def generate_data_quality_report(univariate_results, bivariate_results,
                                  multivariate_results, network_results,
                                  df_shape, missing_report) -> str:
    """Generate data_quality_report.md"""
    lines = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Dataset Shape**: {df_shape[0]} rows x {df_shape[1]} columns")
    lines.append("")
    lines.append("## 1. Univariate Summary")
    lines.append("")
    lines.append("| Column | N | Missing | Mean | Std | Skewness | Best Fit | AIC |")
    lines.append("|--------|---|---------|------|-----|----------|----------|-----|")
    for col, info in univariate_results.items():
        best = info.get('best_fit', {})
        lines.append(f"| {col} | {info['n']} | {info['missing']} | "
                    f"{info['mean']:.3g} | {info['std']:.3g} | "
                    f"{info['skewness']:.3g} | "
                    f"{best.get('distribution', 'N/A')} | "
                    f"{best.get('aic', 'N/A')} |")
    lines.append("")
    lines.append("## 2. Bivariate Relationships")
    lines.append("")
    if bivariate_results and bivariate_results.get('nonlinear_signals'):
        lines.append("### Nonlinear Signals Detected")
        lines.append("")
        for signal in bivariate_results['nonlinear_signals']:
            lines.append(f"- **{signal['pair']}**: Pearson={signal['pearson']:.3f}, "
                        f"Spearman={signal['spearman']:.3f}, "
                        f"Delta={signal['delta']:.3f} -> {signal['interpretation']}")
    lines.append("")
    lines.append("## 3. Multivariate Structure")
    lines.append("")
    if multivariate_results:
        lines.append(f"- **Components for 80% variance**: {multivariate_results['n_components_80pct']}")
        lines.append(f"- **Top PC1 features**: {', '.join(multivariate_results['top_pc1_features'][:5])}")
        lines.append(f"- **Explained variance (PC1-5)**: {[f'{x:.1%}' for x in multivariate_results['explained_variance_ratio']]}")
    lines.append("")
    lines.append("## 4. Network Characteristics")
    lines.append("")
    if network_results:
        for k, v in network_results.items():
            lines.append(f"- **{k}**: {v}")
    lines.append("")
    return '\n'.join(lines)

def run_eda_pipeline(input_path, output_dir='结果/'):
    """Execute the full 4-layer EDA pipeline"""
    print("=" * 60)
    print("深度 EDA 四层分析管线")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    # Load data
    print(f"\n[加载] {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"  形状: {df.shape[0]} 行 x {df.shape[1]} 列")

    # Layer 1: Univariate
    print("\n[1/4] 单变量分布分析...")
    uni_results = univariate_analysis(df)
    n_plots = len(uni_results)
    print(f"  完成: {n_plots} 列, 生成 {n_plots} 张分布图")

    # Layer 2: Bivariate
    print("\n[2/4] 双变量关系分析...")
    bi_results = bivariate_analysis(df)
    n_nonlinear = len(bi_results.get('nonlinear_signals', [])) if bi_results else 0
    print(f"  完成: 发现 {n_nonlinear} 个非线性信号")

    # Layer 3: Multivariate
    print("\n[3/4] 多变量结构分析...")
    multi_results = multivariate_analysis(df)
    if multi_results:
        print(f"  完成: {multi_results['n_components_80pct']} 个主成分覆盖 80% 方差")

    # Layer 4: Network
    print("\n[4/4] 网络特性分析...")
    net_results = network_analysis(df)
    if net_results:
        print(f"  完成: {net_results.get('n_nodes', 'N/A')} 节点, "
              f"{net_results.get('n_edges', 'N/A')} 边")

    # Generate report
    missing_report = pd.DataFrame([
        {'column': c, 'missing_count': int(df[c].isna().sum()),
         'missing_rate': f"{df[c].isna().sum()/len(df):.2%}"}
        for c in df.columns if df[c].isna().sum() > 0
    ])
    report = generate_data_quality_report(
        uni_results, bi_results, multi_results, net_results,
        df.shape, missing_report
    )

    report_path = os.path.join(output_dir, 'data_quality_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ data_quality_report.md -> {report_path}")
    print(f"✅ EDA 图表 -> {output_dir}figures/")
    return {
        'n_univariate_plots': n_plots,
        'n_nonlinear_signals': n_nonlinear,
        'report_path': report_path
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='深度 EDA 四层分析管线')
    parser.add_argument('--input', required=True, help='预处理后数据文件路径')
    parser.add_argument('--output', default='结果/', help='输出目录')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        import sys; sys.exit(1)

    summary = run_eda_pipeline(args.input, args.output)
    print(f"\n管线完成: {summary}")
