# -*- coding: utf-8 -*-
"""
utils.py — 数学建模公共工具函数
用途：全局绘图风格、数据加载、文件保存、置信区间、统计检验辅助
各问题脚本通过 from utils import * 调用
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats

# ============================================================
# 随机种子（可复现——仅使用 default_rng，禁止全局 np.random.seed）
# ============================================================
RNG = np.random.default_rng(42)

# ============================================================
# 路径常量（基于 __file__ 自动定位，无需手动修改）
# ============================================================
_BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = _BASE_DIR / '题目' / '附件数据'
OUTPUT_DIR = _BASE_DIR
FIGURES_DIR = OUTPUT_DIR / '结果' / 'figures'
PAPER_FIGURES_DIR = OUTPUT_DIR / '论文' / 'figures'
for d in [FIGURES_DIR, PAPER_FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 色盲友好配色（国际期刊标准 9 色）
# ============================================================
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00',
          '#56B4E9', '#000000', '#F0E442', '#999999']

# ============================================================
# 全局绘图风格（出版级，≥300dpi）
# ============================================================
mpl.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 600,
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'lines.linewidth': 2, 'lines.markersize': 7,
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
    'errorbar.capsize': 3,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# ============================================================
# 文件操作
# ============================================================

def load_data():
    """加载数据，根据实际赛题修改"""
    pass


def save_fig(fig, name, show=False, dpi=600):
    """保存图表到结果目录和论文目录（≥600dpi 冲刺档出版级）

    Args:
        fig: matplotlib figure
        name: 文件名（含扩展名）
        show: 是否显示
        dpi: 分辨率（默认600，冲刺档标准；位图≥600dpi，矢量图自动忽略）
    """
    for d in [FIGURES_DIR, PAPER_FIGURES_DIR]:
        fig.savefig(os.path.join(d, name), dpi=dpi, bbox_inches='tight',
                    facecolor='white')
    if show:
        plt.show()
    plt.close(fig)


def save_csv(df, name, index=False):
    """保存 DataFrame 到 CSV（UTF-8 BOM，Excel 兼容）"""
    path = OUTPUT_DIR / '结果' / name
    df.to_csv(path, index=index, encoding='utf-8-sig')
    print(f"已保存: {path}")
    return path


# ============================================================
# 置信区间
# ============================================================

def confidence_interval(data, confidence=0.95):
    """计算 t 分布置信区间

    Args:
        data: array-like, 样本数据
        confidence: float, 置信水平（默认 0.95）

    Returns:
        (mean, lower, upper): 均值和置信区间上下界
        如果数据不足则返回 (nan, nan, nan)
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=1000, confidence=0.95):
    """Bootstrap 置信区间

    Args:
        data: array-like
        stat_func: callable, 统计量函数（默认 np.mean）
        n_bootstrap: int, 重采样次数
        confidence: float, 置信水平

    Returns:
        (stat_value, lower, upper)
    """
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan
    boot_stats = [stat_func(RNG.choice(data, n, replace=True))
                  for _ in range(n_bootstrap)]
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)
    return stat_func(data), lower, upper


# ============================================================
# 统计标注
# ============================================================

def add_significance_annotation(ax, x1, x2, y, p_value, y_offset=0.05):
    """在图上添加统计显著性标注条（*p<0.05, **p<0.01, ***p<0.001）"""
    y_max = y * (1 + y_offset)
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'
    ax.plot([x1, x1, x2, x2],
            [y_max * 0.98, y_max, y_max, y_max * 0.98],
            lw=1.5, color='black')
    ax.text((x1 + x2) / 2, y_max * 1.01, sig_text,
            ha='center', va='bottom', fontsize=14, fontweight='bold')


# ============================================================
# 数据编码检测
# ============================================================

def detect_encoding(file_path, sample_size=100000):
    """检测文件编码"""
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw = f.read(sample_size)
        result = chardet.detect(raw)
        return result['encoding'], result['confidence']
    except ImportError:
        return None, 0


ENCODINGS_TO_TRY = ['utf-8', 'gb2312', 'gbk', 'gb18030', 'latin1', 'cp1252']


def smart_read_csv(file_path, **kwargs):
    """智能读取 CSV：自动检测编码并加载"""
    enc, conf = detect_encoding(file_path)
    if enc and conf > 0.7:
        try:
            return pd.read_csv(file_path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, Exception):
            pass
    for encoding in ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(file_path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"无法读取文件 {file_path}，已尝试所有常见编码")
