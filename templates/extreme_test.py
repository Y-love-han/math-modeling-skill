# -*- coding: utf-8 -*-
"""
extreme_test.py — 极端情况测试与退化场景验证
用途：对抗性验证阶段（6.5）——测试模型在边界条件、退化场景、噪声干扰下的行为
理论依据：退化验证原则——如果模型在已知正确答案的特例下输出错误，则存在隐藏缺陷
运行方式：python extreme_test.py（阶段 6.5 必须执行，不可跳过）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from utils import (save_fig, COLORS, RNG, OUTPUT_DIR, FIGURES_DIR,
                   confidence_interval, bootstrap_ci)


# ============================================================
# 1. 参数边界值测试
# ============================================================

def boundary_value_test(model_func, param_ranges, n_test=5):
    """测试模型在参数边界值的表现

    对每个参数，分别在最小值、最大值、0（如适用）、负值（如适用）、
    极大值（如适用）处运行模型，检查是否崩溃、输出是否在合理范围。

    Args:
        model_func: callable, model_func(params_dict) -> float
        param_ranges: dict, {param_name: (min, max, typical)}
        n_test: int, 边界值采样数（默认5：min, max, 0, -inf近似, +inf近似）

    Returns:
        dict: {param_name: [{value, output, status, issue}]}
    """
    results = {}
    for name, (lo, hi, typical) in param_ranges.items():
        param_results = []
        # 测试值列表：最小值、最大值、0、典型值×(-1)（负值）、典型值×100（极大）
        test_values = [lo, hi]
        if lo <= 0 <= hi:
            test_values.append(0.0)
        if typical > 0:
            test_values.append(-typical)
        test_values.append(typical * 100.0)

        for val in test_values:
            params = {k: v[2] for k, v in param_ranges.items()}  # 用典型值
            params[name] = val
            try:
                output = model_func(params)
                issue = None
                # 检查输出合理性
                if np.isnan(output):
                    issue = f"输出为 NaN (参数 {name}={val:.3g})"
                elif np.isinf(output):
                    issue = f"输出为 ∞ (参数 {name}={val:.3g})"
                elif abs(output) > 1e10 and typical < 1e6:
                    issue = f"输出异常巨大 {output:.3g} (参数 {name}={val:.3g})"
                param_results.append({
                    'param_value': val, 'output': output,
                    'status': 'PASS' if issue is None else 'FAIL',
                    'issue': issue
                })
            except Exception as e:
                param_results.append({
                    'param_value': val, 'output': None,
                    'status': 'CRASH',
                    'issue': f"模型崩溃: {str(e)}"
                })
        results[name] = param_results
    return results


# ============================================================
# 2. 退化场景验证（防止"能跑但错"）
# ============================================================

def degenerate_case_test(model_func, standard_func, test_cases):
    """退化场景验证：在已知正确答案的特例下验证模型

    核心思想：如果模型在简化到教科书情形时输出错误，
    则说明模型存在隐藏的逻辑缺陷，即使复杂情形下"看起来对"。

    Args:
        model_func: callable, 待验证模型 params -> output
        standard_func: callable, 已知正确结果 params -> ground_truth
        test_cases: list of dict, [{name, params, description}]

    Returns:
        list: [{name, description, expected, actual, rel_error, status}]
    """
    results = []
    for case in test_cases:
        try:
            expected = standard_func(case['params'])
            actual = model_func(case['params'])
            rel_error = abs(actual - expected) / (abs(expected) + 1e-12)
            results.append({
                'name': case['name'],
                'description': case['description'],
                'expected': expected,
                'actual': actual,
                'rel_error': rel_error,
                'status': 'PASS' if rel_error < 0.01 else 'FAIL'
            })
        except Exception as e:
            results.append({
                'name': case['name'],
                'description': case.get('description', ''),
                'expected': None,
                'actual': None,
                'rel_error': None,
                'status': 'CRASH',
                'issue': str(e)
            })
    return results


# ============================================================
# 3. 噪声鲁棒性测试
# ============================================================

def noise_robustness_test(model_func, base_params, noise_levels=(0.05, 0.10, 0.20),
                          n_repeats=100):
    """测试模型在输入数据加噪声后的鲁棒性

    Args:
        model_func: callable, params -> output
        base_params: dict, 基准参数
        noise_levels: tuple, 噪声水平（比例，如 5%/10%/20%）
        n_repeats: int, 每个噪声水平的重复次数

    Returns:
        dict: {noise_level: {mean, std, ci_95, degradation_pct}}
    """
    base_output = model_func(base_params)
    results = {}

    for level in noise_levels:
        outputs = []
        for _ in range(n_repeats):
            noisy_params = base_params.copy()
            for k, v in noisy_params.items():
                if isinstance(v, (int, float)) and v != 0:
                    noise = RNG.normal(0, abs(v) * level)
                    noisy_params[k] = v + noise
            try:
                outputs.append(model_func(noisy_params))
            except Exception:
                pass

        if outputs:
            outputs = np.array(outputs)
            mean, ci_low, ci_high = confidence_interval(outputs)
            degradation = abs(mean - base_output) / (abs(base_output) + 1e-12)
            results[f'{int(level * 100)}%噪声'] = {
                'mean': mean, 'std': np.std(outputs),
                'ci_95': (ci_low, ci_high),
                'degradation_pct': degradation * 100,
                'n_valid': len(outputs)
            }
    return results


# ============================================================
# 4. 备选方案对比测试
# ============================================================

def alternative_method_test(primary_func, alternative_funcs, test_inputs):
    """对比主要方法 vs 备选方法的性能差异

    Args:
        primary_func: callable, (params) -> output
        alternative_funcs: dict, {name: callable}
        test_inputs: list of dict, 测试输入列表

    Returns:
        dict: {method_name: {outputs[], mean, std, vs_primary_pct}}
    """
    all_outputs = {'primary': [primary_func(inp) for inp in test_inputs]}
    primary_mean = np.mean(all_outputs['primary'])

    for name, func in alternative_funcs.items():
        outputs = [func(inp) for inp in test_inputs]
        all_outputs[name] = outputs

    results = {}
    for name, outputs in all_outputs.items():
        out_arr = np.array(outputs)
        diff_pct = ((np.mean(out_arr) - primary_mean) /
                     (abs(primary_mean) + 1e-12) * 100)
        results[name] = {
            'mean': np.mean(out_arr),
            'std': np.std(out_arr),
            'vs_primary_pct': diff_pct
        }
    return results


# ============================================================
# 5. 可视化
# ============================================================

def plot_boundary_test_results(boundary_results):
    """绘制边界值测试结果热力图（哪些参数-边界组合失败）"""
    params = list(boundary_results.keys())
    status_map = {'PASS': 1, 'FAIL': 0, 'CRASH': -1}

    fig, axes = plt.subplots(len(params), 1, figsize=(10, max(4, len(params) * 1.5)),
                             squeeze=False)
    for i, name in enumerate(params):
        ax = axes[i][0]
        values = [r['param_value'] for r in boundary_results[name]]
        outputs = [r.get('output', 0) or 0 for r in boundary_results[name]]
        statuses = [r['status'] for r in boundary_results[name]]
        colors = [{'PASS': COLORS[2], 'FAIL': COLORS[3], 'CRASH': COLORS[4]}[s]
                  for s in statuses]
        ax.scatter(values, outputs, c=colors, s=100, edgecolors='black', zorder=5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel(f'{name}\n输出')
        ax.grid(True, alpha=0.3)

        # 标注失败项
        for r in boundary_results[name]:
            if r['status'] != 'PASS':
                ax.annotate(r['issue'][:40], (r['param_value'], r.get('output', 0) or 0),
                           fontsize=7, color='red', xytext=(5, 5),
                           textcoords='offset points')
    axes[-1][0].set_xlabel('参数值')
    fig.suptitle('边界值测试结果（🟢PASS 🔴FAIL ⚫CRASH）', fontsize=14)
    plt.tight_layout()
    return fig


def plot_noise_degradation(noise_results):
    """绘制噪声退化曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    levels = list(noise_results.keys())
    degs = [noise_results[l]['degradation_pct'] for l in levels]
    ax.plot(levels, degs, 'o-', color=COLORS[0], linewidth=2, markersize=10)
    ax.axhline(5, color=COLORS[3], linestyle='--', label='5% 退化警戒线')
    ax.axhline(10, color=COLORS[4], linestyle='--', label='10% 退化红线')
    ax.set_xlabel('噪声水平')
    ax.set_ylabel('输出退化百分比 (%)')
    ax.set_title('噪声鲁棒性测试')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================
# 6. 综合测试报告
# ============================================================

def generate_extreme_test_report(boundary, degenerate, noise, alternative):
    """生成极端情况测试综合报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("          极端情况与对抗性测试综合报告")
    lines.append("=" * 70)

    # 边界测试
    lines.append("\n【1. 参数边界值测试】")
    for name, cases in boundary.items():
        fails = [c for c in cases if c['status'] != 'PASS']
        lines.append(f"  {name}: {len(cases)} 个测试, "
                    f"失败 {len(fails)} ({'✅' if not fails else '❌'})")
        for f in fails:
            lines.append(f"    → {f['issue']}")

    # 退化场景
    lines.append("\n【2. 退化场景验证】")
    if degenerate:
        fails = [d for d in degenerate if d['status'] != 'PASS']
        lines.append(f"  共 {len(degenerate)} 个退化场景, "
                    f"失败 {len(fails)} ({'✅' if not fails else '❌'})")
        for d in degenerate:
            icon = '✅' if d['status'] == 'PASS' else '❌'
            lines.append(f"  {icon} {d['name']}: 期望={d.get('expected','N/A')}, "
                        f"实际={d.get('actual','N/A')}, "
                        f"相对误差={d.get('rel_error','N/A')}")
    else:
        lines.append("  ⚠️ 未配置退化场景测试用例")

    # 噪声
    lines.append("\n【3. 噪声鲁棒性测试】")
    for level, stats in noise.items():
        flag = '✅' if stats['degradation_pct'] < 10 else '❌'
        lines.append(f"  {flag} {level}: 退化 {stats['degradation_pct']:.2f}%, "
                    f"95%CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")

    # 备选方案
    lines.append("\n【4. 备选方案对比】")
    if alternative:
        for name, stats in alternative.items():
            diff = stats['vs_primary_pct']
            direction = '优于' if diff > 0 else '劣于'
            lines.append(f"  {name}: 均值={stats['mean']:.4f}, "
                        f"{direction}主方法 {abs(diff):.1f}%")
    else:
        lines.append("  ⚠️ 未配置备选方案")

    lines.append("\n" + "=" * 70)
    return '\n'.join(lines)


# ============================================================
# 示例代码（执行时替换为实际模型）
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("极端情况测试与退化场景验证")
    print("=" * 60)

    # === 示例：配置测试 ===
    # 1) 定义模型函数
    def example_model(params):
        """示例模型（替换为实际模型）"""
        a, b, c = params['a'], params['b'], params['c']
        if a < 0:
            raise ValueError("参数 a 不能为负")
        return a * np.sin(b) + np.exp(-c * a)

    # 2) 定义参数范围
    example_param_ranges = {
        'a': (0.01, 10.0, 1.0),
        'b': (0.0, 2 * np.pi, 1.0),
        'c': (-5.0, 5.0, 0.5),
    }

    # 3) 定义退化场景
    def degenerate_standard(params):
        """退化场景的标准结果（替换为已知正确结果）"""
        # 如：当 a→0 时，模型应退化为 1 + b (一阶 Taylor)
        a, b, c = params['a'], params['b'], params['c']
        if a < 0.001:
            return 1.0 + a * (np.sin(b) - c)  # 一阶近似
        return a * np.sin(b) + np.exp(-c * a)

    test_cases = [
        {
            'name': 'a→0 退化',
            'description': '当 a→0 时, exp(-ca) ≈ 1-ca, 模型应退化为一阶近似',
            'params': {'a': 0.001, 'b': 1.0, 'c': 0.5}
        },
        {
            'name': '对称情况 b=π',
            'description': 'b=π 时 sin(b)=0, 模型简化为 exp(-ca)',
            'params': {'a': 1.0, 'b': np.pi, 'c': 0.5}
        },
    ]

    # === 执行测试 ===
    print("\n[1/4] 参数边界值测试...")
    boundary_results = boundary_value_test(example_model, example_param_ranges)
    n_fails = sum(1 for cases in boundary_results.values()
                  for c in cases if c['status'] != 'PASS')
    print(f"  完成: {n_fails} 个失败")

    print("\n[2/4] 退化场景验证...")
    degen_results = degenerate_case_test(example_model, degenerate_standard,
                                          test_cases)
    n_degen_fails = sum(1 for d in degen_results if d['status'] != 'PASS')
    print(f"  完成: {n_degen_fails} 个失败")

    print("\n[3/4] 噪声鲁棒性测试...")
    noise_results = noise_robustness_test(example_model,
                                           {'a': 1.0, 'b': 1.0, 'c': 0.5})
    for level, stats in noise_results.items():
        print(f"  {level}: 退化 {stats['degradation_pct']:.2f}%")

    print("\n[4/4] 备选方案对比...")
    # 示例备选方案
    alt_funcs = {
        '简化模型(无exp项)': lambda p: p['a'] * np.sin(p['b']) + 1.0,
        '线性近似': lambda p: p['a'] * p['b'] + 1.0 - p['c'] * p['a'],
    }
    test_inputs = [
        {'a': 0.5, 'b': 0.5, 'c': 1.0},
        {'a': 1.0, 'b': 1.0, 'c': 0.5},
        {'a': 2.0, 'b': 1.5, 'c': 0.2},
    ]
    alt_results = alternative_method_test(example_model, alt_funcs, test_inputs)
    for name, stats in alt_results.items():
        print(f"  {name}: vs 主方法 {stats['vs_primary_pct']:+.1f}%")

    # === 生成报告 ===
    print("\n生成测试报告...")
    report = generate_extreme_test_report(boundary_results, degen_results,
                                           noise_results, alt_results)
    print(report)

    # 保存报告
    report_path = OUTPUT_DIR / '结果' / 'extreme_test_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存至: {report_path}")

    # 生成图片
    fig1 = plot_boundary_test_results(boundary_results)
    save_fig(fig1, 'figE1_boundary_test.png')
    fig2 = plot_noise_degradation(noise_results)
    save_fig(fig2, 'figE2_noise_degradation.png')
    print("图表已保存")

    # 退出码：全部通过=0，有失败=1
    import sys
    sys.exit(0 if n_fails == 0 and n_degen_fails == 0 else 1)
