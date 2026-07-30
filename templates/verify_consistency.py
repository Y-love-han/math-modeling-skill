# -*- coding: utf-8 -*-
"""
verify_consistency.py — 代码-论文一致性自动验证脚本
用途：扫描论文 LaTeX 源码中的所有数值，与代码输出日志交叉验证
      检查图表引用完整性、代码可运行性、公式标签一致性、参考文献可追溯性
运行方式：python verify_consistency.py --tex main.tex --code_dir 代码/ --log_dir 结果/
输出：consistency_report.txt（5 类验证总报告）
"""
import re
import os
import csv
import json
import ast
import sys
from pathlib import Path
from typing import Dict, List


class ConsistencyVerifier:
    """代码-论文一致性验证器（5 类验证）"""

    def __init__(self, tex_path: str, code_dir: str, log_dir: str):
        self.tex_path = Path(tex_path)
        self.code_dir = Path(code_dir)
        self.log_dir = Path(log_dir)
        self.tex_content = self._read(self.tex_path)
        self.results = {}

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return open(path, 'r', encoding='utf-8').read()
        except FileNotFoundError:
            return ''

    # ================================================================
    # V1: 数值结果一致性
    # ================================================================
    def verify_numerical(self, rel_tol=0.01, abs_tol=1e-9) -> Dict:
        """验证论文数值是否在代码输出中找到匹配"""
        issues, matched, unmatched = [], 0, []

        # 从 LaTeX 提取数值
        tex_nums = []
        # 跳过注释行，匹配 ≥4 位的整数、≥2 位小数的浮点、科学计数法
        content = '\n'.join(l for l in self.tex_content.split('\n')
                            if not l.strip().startswith('%'))
        pattern = r'(?<![\d.])(\d+\.\d{2,}(?:[eE][+-]?\d+)?|\d{4,}(?:[eE][+-]?\d+)?)(?![\d.])'
        for m in re.finditer(pattern, content):
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            ctx_start = max(0, m.start() - 40)
            ctx_end = min(len(content), m.end() + 40)
            tex_nums.append({
                'value': val, 'raw': m.group(1),
                'context': content[ctx_start:ctx_end].replace('\n', ' ').strip(),
                'line': content[:m.start()].count('\n') + 1
            })

        # 从代码输出加载数值
        code_nums = []
        for csv_f in self.log_dir.glob('*.csv'):
            try:
                for row in csv.reader(open(csv_f, 'r', encoding='utf-8')):
                    for cell in row:
                        try:
                            code_nums.append({'value': float(cell), 'source': csv_f.name})
                        except ValueError:
                            pass
            except Exception:
                pass
        for txt_f in self.log_dir.glob('*.txt'):
            try:
                for line in open(txt_f, 'r', encoding='utf-8'):
                    for m in re.finditer(r'\d+\.\d{2,}(?:[eE][+-]?\d+)?|\d{4,}(?:[eE][+-]?\d+)?', line):
                        try:
                            code_nums.append({'value': float(m.group()), 'source': txt_f.name})
                        except ValueError:
                            pass
            except Exception:
                pass

        # 交叉匹配
        significant = [n for n in tex_nums if abs(n['value']) >= 0.001 or len(n['raw']) >= 4]
        for tn in significant:
            found = False
            for cn in code_nums:
                diff = abs(tn['value'] - cn['value'])
                threshold = max(rel_tol * max(abs(tn['value']), abs(cn['value'])), abs_tol)
                if diff <= threshold:
                    matched += 1
                    found = True
                    break
            if not found:
                unmatched.append(tn)

        total = len(significant)
        match_rate = matched / total if total > 0 else 0
        for item in unmatched[:20]:
            issues.append(f"行{item.get('line','?')}: {item['raw']} 未在代码输出中找到匹配 ({item.get('context','')[:60]})")

        self.results['numerical'] = {
            'status': len(unmatched) == 0 and match_rate >= 0.8,
            'issues': issues,
            'stats': {'total': total, 'matched': matched, 'unmatched': len(unmatched), 'match_rate': f'{match_rate:.1%}'}
        }
        return self.results['numerical']

    # ================================================================
    # V2: 图表数据一致性
    # ================================================================
    def verify_figure_table(self) -> Dict:
        """验证图表引用完整性"""
        issues = []
        fig_refs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', self.tex_content)
        figures_dir = self.tex_path.parent / 'figures'

        for fig_path in fig_refs:
            full = figures_dir / fig_path
            found = full.exists()
            if not found:
                for ext in ['.png', '.pdf', '.jpg', '.jpeg', '.eps']:
                    if (figures_dir / f'{fig_path}{ext}').exists():
                        found = True
                        break
            if not found:
                issues.append(f"图表文件不存在: {fig_path}")

        all_labels = set(re.findall(r'\\label\{([^}]+)\}', self.tex_content))
        all_refs = re.findall(r'\\ref\{([^}]+)\}', self.tex_content)
        for ref in all_refs:
            if ref not in all_labels:
                issues.append(f"引用 \\ref{{{ref}}} 无对应 label")

        self.results['figure_table'] = {
            'status': len(issues) == 0,
            'issues': issues,
            'stats': {'fig_refs': len(fig_refs), 'labels': len(all_labels), 'unresolved_refs': len([r for r in all_refs if r not in all_labels])}
        }
        return self.results['figure_table']

    # ================================================================
    # V3: 代码完整性验证
    # ================================================================
    def verify_code_integrity(self) -> Dict:
        """验证代码文件语法和完整性"""
        issues = []
        py_files = list(self.code_dir.glob('*.py'))
        if not py_files:
            issues.append("代码目录中未找到 .py 文件")

        stdlib = {'os', 'sys', 're', 'json', 'math', 'datetime', 'pathlib', 'typing',
                  'collections', 'itertools', 'functools', 'warnings', 'subprocess', 'csv', 'ast'}

        for py_f in py_files:
            content = self._read(py_f)
            if not content:
                issues.append(f"代码文件为空: {py_f.name}")
                continue
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append(f"{py_f.name}: 语法错误 行{e.lineno}: {e.msg}")

            if '__main__' not in content and py_f.name not in ('utils.py', 'requirements.txt'):
                issues.append(f"{py_f.name}: 缺少 if __name__ == '__main__' 入口")

            imports = set()
            for m in re.finditer(r'(?:^|\n)\s*(?:import|from)\s+([\w.]+)', content):
                imports.add(m.group(1).split('.')[0])
            req_path = self.code_dir / 'requirements.txt'
            req_content = self._read(req_path).lower() if req_path.exists() else ''
            pkg_map = {'sklearn': 'scikit-learn', 'cv2': 'opencv-python', 'PIL': 'Pillow', 'yaml': 'PyYAML'}
            for imp in imports - stdlib:
                pkg = pkg_map.get(imp, imp).lower()
                if pkg not in req_content:
                    issues.append(f"{py_f.name}: import '{imp}' 未在 requirements.txt 中声明")

        if not (self.code_dir / 'requirements.txt').exists():
            issues.append("缺少 requirements.txt")

        self.results['code_integrity'] = {
            'status': len(issues) == 0,
            'issues': issues,
            'stats': {'code_files': len(py_files), 'files': [f.name for f in py_files]}
        }
        return self.results['code_integrity']

    # ================================================================
    # V4: 公式一致性
    # ================================================================
    def verify_formula(self) -> Dict:
        """验证公式标签和引用一致性"""
        issues = []
        eq_labels = set(re.findall(r'\\label\{(eq:[^}]+)\}', self.tex_content))
        eq_refs = re.findall(r'\\(?:ref|eqref|cref)\{(eq:[^}]+)\}', self.tex_content)
        equations = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', self.tex_content, re.DOTALL)

        unlabeled = sum(1 for eq in equations if '\\label' not in eq)
        if unlabeled:
            issues.append(f"{unlabeled} 个编号公式缺少 \\label")

        for ref in eq_refs:
            if ref not in eq_labels:
                issues.append(f"公式引用 {ref} 无对应 label")

        for label in eq_labels:
            if not re.search(r'\\(?:ref|eqref|cref)\{' + re.escape(label) + r'\}', self.tex_content):
                issues.append(f"公式 label {label} 已定义但未引用")

        self.results['formula'] = {
            'status': len(issues) == 0,
            'issues': issues,
            'stats': {'total_eqs': len(equations), 'unlabeled': unlabeled, 'labels': len(eq_labels), 'refs': len(eq_refs)}
        }
        return self.results['formula']

    # ================================================================
    # V5: 参考文献可追溯性
    # ================================================================
    def verify_reference(self) -> Dict:
        """验证参考文献引用和被引一致性"""
        issues = []
        bibitems = re.findall(r'\\bibitem\{([^}]+)\}', self.tex_content)
        bib_set = set(bibitems)
        cite_keys = set()
        for m in re.findall(r'\\(?:cite|upcite|citep|citet)\{([^}]+)\}', self.tex_content):
            for key in m.split(','):
                cite_keys.add(key.strip())

        for key in cite_keys - bib_set:
            issues.append(f"引用 {key} 在参考文献列表中未找到")
        for key in bib_set - cite_keys:
            issues.append(f"参考文献 {key} 在正文中未被引用")
        if len(bibitems) < 15:
            issues.append(f"参考文献仅 {len(bibitems)} 篇，少于 15 篇要求")

        en_count = 0
        for m in re.finditer(r'\\bibitem\{[^}]+\}\s*(.*?)(?=\\bibitem|$)', self.tex_content, re.DOTALL):
            entry = m.group(1).strip()
            ascii_letters = len(re.findall(r'[a-zA-Z]', entry))
            chinese = len(re.findall(r'[\u4e00-\u9fff]', entry))
            if (ascii_letters + chinese) > 0 and ascii_letters / (ascii_letters + chinese) > 0.5:
                if len(re.findall(r'[a-zA-Z]{4,}', entry)) >= 3:
                    en_count += 1

        if en_count < 5:
            issues.append(f"英文文献仅 {en_count} 篇，少于 5 篇要求")

        pending = re.findall(r'待验证|待查证|DOI待验证', self.tex_content)
        if pending:
            issues.append(f"发现 {len(pending)} 处未处理的'待验证'标注")

        self.results['reference'] = {
            'status': len(issues) == 0,
            'issues': issues,
            'stats': {'total': len(bibitems), 'cited': len(cite_keys & bib_set),
                      'uncited': len(bib_set - cite_keys), 'unresolved': len(cite_keys - bib_set),
                      'en_refs': en_count, 'pending': len(pending)}
        }
        return self.results['reference']

    # ================================================================
    # 执行全部验证并生成报告
    # ================================================================
    def run_all(self) -> Dict:
        print("正在执行代码-论文一致性验证...")
        print(f"  论文: {self.tex_path}")
        print(f"  代码: {self.code_dir}")
        print(f"  日志: {self.log_dir}\n")

        for i, (name, method) in enumerate([
            ('数值结果一致性', self.verify_numerical),
            ('图表数据一致性', self.verify_figure_table),
            ('代码完整性', self.verify_code_integrity),
            ('公式一致性', self.verify_formula),
            ('参考文献可追溯性', self.verify_reference),
        ], 1):
            result = method()
            icon = '✅' if result['status'] else '❌'
            print(f"  [{i}/5] {name}: {icon}")
        print()
        return self.results

    def generate_report(self) -> str:
        lines = ["=" * 70, "          代码-论文一致性验证总报告", "=" * 70, ""]
        lines.append(f"{'验证项':<20} {'状态':<10} {'问题数':<10}")
        lines.append("-" * 70)
        all_pass = True
        for key, label in [
            ('numerical', '数值结果一致性'),
            ('figure_table', '图表数据一致性'),
            ('code_integrity', '代码完整性'),
            ('formula', '公式一致性'),
            ('reference', '参考文献可追溯性'),
        ]:
            r = self.results.get(key, {})
            status = '✅ 通过' if r.get('status') else '❌ 失败'
            if not r.get('status'):
                all_pass = False
            n = len(r.get('issues', []))
            lines.append(f"{label:<18} {status:<10} {n:<10}")
        lines.append("-" * 70)

        for key, label in [
            ('numerical', '数值结果一致性'),
            ('figure_table', '图表数据一致性'),
            ('code_integrity', '代码完整性'),
            ('formula', '公式一致性'),
            ('reference', '参考文献可追溯性'),
        ]:
            r = self.results.get(key, {})
            if r.get('issues'):
                lines.append(f"\n【{label}】问题详情：")
                for i, issue in enumerate(r['issues'], 1):
                    lines.append(f"  {i}. {issue}")

        lines.append(f"\n{'=' * 70}")
        if all_pass:
            lines.append("✅ 一致性验证全部通过，可以提交")
        else:
            failed = [k for k, v in self.results.items() if not v.get('status')]
            lines.append(f"⛔ {len(failed)} 项验证失败，禁止提交！失败项: {', '.join(failed)}")
        lines.append("=" * 70)
        return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='代码-论文一致性验证')
    parser.add_argument('--tex', required=True, help='LaTeX 论文文件路径')
    parser.add_argument('--code_dir', required=True, help='代码目录路径')
    parser.add_argument('--log_dir', required=True, help='代码输出日志目录')
    args = parser.parse_args()

    verifier = ConsistencyVerifier(args.tex, args.code_dir, args.log_dir)
    verifier.run_all()
    report = verifier.generate_report()
    print(report)

    report_path = Path(args.log_dir) / 'consistency_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存至: {report_path}")

    all_pass = all(v.get('status') for v in verifier.results.values())
    sys.exit(0 if all_pass else 1)
