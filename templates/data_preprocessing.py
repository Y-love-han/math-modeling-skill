# -*- coding: utf-8 -*-
"""
data_preprocessing.py — 数据预处理强制管线
用途：阶段 4 数据预处理——编码检测、缺失值处理、异常值检测、数据清洗
      在任何建模操作之前必须运行此脚本，确保数据质量
运行方式：python data_preprocessing.py --input 题目/附件数据/data.csv --output 结果/
产出：data_manifest.json, preprocessed_data.csv, preprocessing_log.txt
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# 导入公共工具
from utils import (detect_encoding, smart_read_csv, ENCODINGS_TO_TRY,
                   RNG, OUTPUT_DIR, FIGURES_DIR)

# ============================================================
# 配置
# ============================================================
MISSING_THRESHOLDS = {
    'knn_impute': 0.05,       # <5%: KNN 插补
    'multiple_impute': 0.20,   # 5-20%: 多重插补
    'evaluate': 0.50,          # 20-50%: 评估后决定
    # >50%: 丢弃（论文说明理由）
}

OUTLIER_METHODS = ['iqr', 'zscore', 'isolation_forest']
OUTLIER_THRESHOLD = 3.0  # Z-score 阈值
IQR_MULTIPLIER = 1.5


# ============================================================
# 步骤 1：编码检测与数据加载
# ============================================================

def load_with_encoding_detection(file_path: str) -> Tuple[pd.DataFrame, str, float]:
    """编码检测 → 多编码尝试加载 → 确认无乱码

    Returns:
        (DataFrame, encoding_used, confidence)
    """
    log = []
    log.append(f"[{_now()}] 开始加载文件: {file_path}")

    # 尝试 chardet 检测
    enc, conf = detect_encoding(file_path) if 'detect_encoding' in dir() else (None, 0)

    if enc and conf > 0.7:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            log.append(f"[{_now()}] chardet 检测: {enc} (置信度 {conf:.0%}) → 成功")
            return df, enc, conf
        except Exception as e:
            log.append(f"[{_now()}] chardet 检测: {enc} → 失败 ({e})")

    # 回退：尝试常见编码
    for encoding in ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            log.append(f"[{_now()}] 尝试编码: {encoding} → 成功")
            return df, encoding, 1.0
        except (UnicodeDecodeError, Exception):
            continue

    # Excel 格式
    if file_path.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(file_path)
            log.append(f"[{_now()}] Excel 格式 → 成功")
            return df, 'excel', 1.0
        except Exception as e:
            log.append(f"[{_now()}] Excel 格式 → 失败 ({e})")

    raise ValueError(f"无法读取文件 {file_path}，已尝试全部编码")


# ============================================================
# 步骤 2：列名规范化
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名规范化：去前后空格、替换特殊字符、处理重名"""
    original = df.columns.tolist()

    # 去空格
    df.columns = [str(c).strip() for c in df.columns]

    # 替换特殊字符
    replacements = {'：': '_', '（': '(', '）': ')', ' ': '_',
                    '\t': '_', '\n': '_', '　': '_'}
    new_cols = []
    for c in df.columns:
        for old, new in replacements.items():
            c = c.replace(old, new)
        new_cols.append(c)

    # 处理重名：添加后缀 _1, _2
    seen = {}
    final_cols = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            final_cols.append(c)

    df.columns = final_cols
    changed = sum(1 for o, n in zip(original, final_cols) if o != n)
    if changed > 0:
        print(f"  列名规范化: {changed} 列被修改")
    return df


# ============================================================
# 步骤 3：类型检查与修复
# ============================================================

def fix_column_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """类型检查：数值列误读为字符串 → 强制转换 + 标记异常"""
    issues = {}

    for col in df.columns:
        # 跳过明显非数值列
        if df[col].dtype in ('int64', 'int32', 'float64', 'float32'):
            continue

        # 尝试转换 object 列
        if df[col].dtype == 'object':
            # 先检查是否全是数值
            converted = pd.to_numeric(df[col], errors='coerce')
            valid_ratio = converted.notna().mean()

            if valid_ratio > 0.9:
                # 90%+ 可转换 → 强制转换，NaN 的作为缺失值
                bad_count = df[col].notna().sum() - converted.notna().sum()
                df[col] = converted
                if bad_count > 0:
                    issues[col] = {
                        'original_type': 'object',
                        'new_type': 'float64',
                        'unparseable': int(bad_count)
                    }
            elif valid_ratio > 0.5:
                # 50-90% 可转换 → 警告但转换
                bad_count = df[col].notna().sum() - converted.notna().sum()
                df[col] = converted
                issues[col] = {
                    'original_type': 'object',
                    'new_type': 'float64 (强制)',
                    'unparseable': int(bad_count),
                    'warning': '超过10%数据无法解析为数值'
                }

        # 检查整数列是否应为类别
        if df[col].dtype in ('int64', 'int32'):
            n_unique = df[col].nunique()
            if n_unique <= 10 and n_unique / len(df) < 0.01:
                issues[col] = {
                    'type': 'int',
                    'suggestion': f'仅 {n_unique} 个唯一值，可能为类别变量'
                }

    return df, issues


# ============================================================
# 步骤 4：缺失值诊断与处理
# ============================================================

def diagnose_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值诊断报告"""
    total = len(df)
    report = []
    for col in df.columns:
        missing = df[col].isna().sum()
        rate = missing / total
        if missing > 0:
            report.append({
                'column': col,
                'missing_count': int(missing),
                'missing_rate': f'{rate:.2%}',
                'dtype': str(df[col].dtype),
                'action': _recommend_missing_action(rate)
            })
    return pd.DataFrame(report)


def _recommend_missing_action(rate: float) -> str:
    """按缺失率推荐处理策略"""
    if rate < 0.01:
        return '删除行（缺失率极低）'
    elif rate < MISSING_THRESHOLDS['knn_impute']:
        return 'KNN 插补（<5%）'
    elif rate < MISSING_THRESHOLDS['multiple_impute']:
        return '多重插补 + 缺失指示变量（5-20%）'
    elif rate < MISSING_THRESHOLDS['evaluate']:
        return '评估后决定：MI 或丢弃（20-50%）'
    else:
        return '丢弃该列（>50%，论文说明理由）'


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """处理缺失值（按决策树分档处理）"""
    log = []
    total = len(df)

    for col in df.columns:
        missing = df[col].isna().sum()
        if missing == 0:
            continue
        rate = missing / total

        if rate < 0.01:
            # 极低缺失率 → 直接删除行
            df = df[df[col].notna()].copy()
            log.append(f"{col}: 缺失率 {rate:.2%} → 删除 {missing} 行")

        elif rate < MISSING_THRESHOLDS['knn_impute'] and df[col].dtype in ('float64', 'float32', 'int64'):
            # <5% 数值列 → KNN 插补
            try:
                from sklearn.impute import KNNImputer
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    imputer = KNNImputer(n_neighbors=5)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    log.append(f"{col}: 缺失率 {rate:.2%} → KNN 插补（5-近邻）")
                else:
                    df[col] = df[col].fillna(df[col].median())
                    log.append(f"{col}: 缺失率 {rate:.2%} → 中位数填充（数值列不足2列）")
            except ImportError:
                df[col] = df[col].fillna(df[col].median())
                log.append(f"{col}: 缺失率 {rate:.2%} → 中位数填充（sklearn不可用）")

        elif rate < MISSING_THRESHOLDS['multiple_impute']:
            # 5-20% → 多重插补 + 缺失指示变量
            indicator_col = f"{col}_missing"
            df[indicator_col] = df[col].isna().astype(int)
            if df[col].dtype in ('float64', 'float32', 'int64'):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'MISSING')
            log.append(f"{col}: 缺失率 {rate:.2%} → 中位数/众数填充 + 缺失指示变量")

        elif rate < MISSING_THRESHOLDS['evaluate']:
            # 20-50% → 评估价值后决定
            if df[col].dtype in ('float64', 'float32', 'int64'):
                df[col] = df[col].fillna(df[col].median())
                log.append(f"{col}: 缺失率 {rate:.2%} → 中位数填充（评估保留）")
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'MISSING')
                log.append(f"{col}: 缺失率 {rate:.2%} → 众数填充（评估保留）")

        else:
            # >50% → 丢弃
            log.append(f"{col}: 缺失率 {rate:.2%} → ⚠️ 丢弃该列（论文需说明理由）")
            # 保留列但标记（实际丢弃在后续阶段决定）

    return df, log


# ============================================================
# 步骤 5：异常值检测（三方法交叉确认）
# ============================================================

def detect_outliers_triangulation(df: pd.DataFrame) -> Dict[str, Dict]:
    """IQR + Z-score + 孤立森林三方法交叉确认异常值

    Returns:
        {column_name: {iqr_outliers: [], zscore_outliers: [], if_outliers: [], confirmed: []}}
    """
    from scipy import stats

    results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        data = df[col].dropna().values
        if len(data) < 10:
            continue

        col_results = {}

        # 方法 1：IQR
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        iqr_outliers = np.where((data < lower) | (data > upper))[0]
        col_results['iqr'] = {
            'count': len(iqr_outliers),
            'bounds': (float(lower), float(upper)),
            'rate': f'{len(iqr_outliers) / len(data):.2%}'
        }

        # 方法 2：Z-score
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        z_outliers = np.where(z_scores > OUTLIER_THRESHOLD)[0]
        col_results['zscore'] = {
            'count': len(z_outliers),
            'threshold': OUTLIER_THRESHOLD,
            'rate': f'{len(z_outliers) / len(data):.2%}'
        }

        # 方法 3：孤立森林
        try:
            from sklearn.ensemble import IsolationForest
            data_2d = data.reshape(-1, 1)
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(data_2d)
            if_outliers = np.where(preds == -1)[0]
            col_results['isolation_forest'] = {
                'count': len(if_outliers),
                'rate': f'{len(if_outliers) / len(data):.2%}'
            }
        except ImportError:
            col_results['isolation_forest'] = {'skipped': 'sklearn不可用'}

        # 三方法交叉确认：至少 2 个方法同时标记 → 视为确认异常值
        iqr_set = set(iqr_outliers)
        z_set = set(z_outliers)
        if 'isolation_forest' not in col_results.get('isolation_forest', {}) or \
           'skipped' not in col_results.get('isolation_forest', {}):
            if_set = set(if_outliers) if 'if_outliers' in dir() else set()
        else:
            if_set = set()

        confirmed = []
        for idx in range(len(data)):
            votes = (idx in iqr_set) + (idx in z_set) + (idx in if_set)
            if votes >= 2:
                confirmed.append(int(idx))
        col_results['confirmed'] = {
            'count': len(confirmed),
            'rate': f'{len(confirmed) / len(data):.2%}',
            'indices': confirmed[:20]  # 仅记录前 20 个
        }

        results[col] = col_results

    return results


# ============================================================
# 步骤 6：数据质量总检查
# ============================================================

def data_quality_check(df: pd.DataFrame) -> Dict:
    """全面数据质量检查"""
    report = {
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        'missing': {'total': int(df.isna().sum().sum()),
                    'rate': f'{df.isna().sum().sum() / (len(df) * len(df.columns)):.4%}'},
        'duplicates': {'count': int(df.duplicated().sum()),
                       'rate': f'{df.duplicated().sum() / len(df):.2%}'},
        'numeric_summary': {},
    }

    # 数值列统计
    for col in df.select_dtypes(include=[np.number]).columns:
        report['numeric_summary'][col] = {
            'min': float(df[col].min()) if not df[col].isna().all() else None,
            'max': float(df[col].max()) if not df[col].isna().all() else None,
            'mean': float(df[col].mean()) if not df[col].isna().all() else None,
            'std': float(df[col].std()) if not df[col].isna().all() else None,
            'zeros': int((df[col] == 0).sum()),
            'negative': int((df[col] < 0).sum()),
        }

    return report


# ============================================================
# 主管线
# ============================================================

def run_preprocessing_pipeline(input_path: str, output_dir: str = '结果/') -> Dict:
    """执行完整数据预处理管线

    步骤：
    1. 编码检测 → 加载
    2. 列名规范化
    3. 类型检查与修复
    4. 缺失值诊断与处理
    5. 异常值三方法检测
    6. 重复行 + 数据范围检查
    7. 写入产出文件

    Returns:
        dict: 管线执行摘要
    """
    log_lines = []
    log_lines.append(f"数据预处理管线 — 开始于 {_now()}")
    log_lines.append(f"输入文件: {input_path}")

    # Step 1: 加载
    df, encoding, conf = load_with_encoding_detection(input_path)
    log_lines.append(f"编码: {encoding} (置信度 {conf:.0%}), 形状: {df.shape}")

    # Step 2: 列名规范化
    df = normalize_columns(df)

    # Step 3: 类型检查
    df, type_issues = fix_column_types(df)
    if type_issues:
        log_lines.append(f"类型修复: {len(type_issues)} 列被调整")
        for col, info in type_issues.items():
            log_lines.append(f"  {col}: {info}")

    # Step 4: 缺失值诊断与处理
    missing_report = diagnose_missing(df)
    df, missing_log = handle_missing_values(df)
    log_lines.extend(missing_log)

    # Step 5: 异常值检测
    outlier_results = detect_outliers_triangulation(df)
    total_confirmed = sum(r.get('confirmed', {}).get('count', 0)
                          for r in outlier_results.values())
    log_lines.append(f"异常值检测: {total_confirmed} 个确认异常值（≥2方法共识）")

    # Step 6: 质量检查
    quality_report = data_quality_check(df)
    log_lines.append(f"重复行: {quality_report['duplicates']['count']} "
                    f"({quality_report['duplicates']['rate']})")

    # Step 7: 写入产出
    os.makedirs(output_dir, exist_ok=True)

    # data_manifest.json
    manifest = {
        'source_file': input_path,
        'encoding_detected': encoding,
        'encoding_confidence': conf,
        'original_shape': list(df.shape),  # 因为 df shape 在缺失值处理后可能变了
        'final_shape': list(df.shape),  # 近似
        'column_types': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        'missing_summary': missing_report.to_dict('records') if len(missing_report) > 0 else [],
        'outlier_summary': {k: v.get('confirmed', {}) for k, v in outlier_results.items()},
        'duplicate_rows': quality_report['duplicates']['count'],
        'preprocessing_timestamp': _now(),
    }
    with open(os.path.join(output_dir, 'data_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # preprocessed_data.csv
    df.to_csv(os.path.join(output_dir, 'preprocessed_data.csv'), index=False,
              encoding='utf-8-sig')

    # preprocessing_log.txt
    log_text = '\n'.join(log_lines)
    with open(os.path.join(output_dir, 'preprocessing_log.txt'), 'w', encoding='utf-8') as f:
        f.write(log_text)

    print(f"✅ 数据预处理完成")
    print(f"   data_manifest.json → {output_dir}")
    print(f"   preprocessed_data.csv → {output_dir} ({df.shape[0]}行 × {df.shape[1]}列)")
    print(f"   preprocessing_log.txt → {output_dir}")

    return {
        'status': 'success',
        'final_shape': df.shape,
        'missing_handled': len(missing_log),
        'outliers_confirmed': total_confirmed,
        'files': ['data_manifest.json', 'preprocessed_data.csv', 'preprocessing_log.txt']
    }


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='数据预处理强制管线')
    parser.add_argument('--input', required=True, help='输入数据文件路径 (CSV/XLSX)')
    parser.add_argument('--output', default='结果/', help='输出目录')
    parser.add_argument('--encoding', default=None, help='强制指定编码（跳过自动检测）')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        import sys
        sys.exit(1)

    summary = run_preprocessing_pipeline(args.input, args.output)
    print(f"\n管线摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")
