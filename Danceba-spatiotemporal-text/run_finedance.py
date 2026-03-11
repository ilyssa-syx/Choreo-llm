"""
遍历多个 ablation 实验的所有 epoch，依次计算 metrics，
每算完一个 epoch 立即写入 results.json，最后绘制折线图。

用法：
    cd /network_space/server126/shared/sunyx/models/Danceba-spatiotemporal-text
    python run_ablation_metrics.py
    # 或指定 gt_root：
    python run_ablation_metrics.py --gt_root data/aist_features_zero_start
    # 或只跑某些 ablation：
    python run_ablation_metrics.py --ablations ablation_time_only ablation_nomask
"""

import os
import sys
import json
import argparse
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 路径设置 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR    = os.path.join(SCRIPT_DIR, 'utils')
EXP_BASE_DIR = os.path.join(SCRIPT_DIR, 'experiments')

# metrics_new.py 内部使用相对 import（features.*），必须从 utils/ 目录导入
sys.path.insert(0, UTILS_DIR)
os.chdir(UTILS_DIR)          # 切换工作目录，保证 features 包可以被找到

import metrics_new_finedance as mn     # noqa: E402  (after chdir)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt_root',
    default=os.path.join(UTILS_DIR, '../data/finedance_features_zero_start'),
    help='Ground-truth feature root (默认 data/finedance_features_zero_start，相对于 utils/)'
)
parser.add_argument(
    '--ablations',
    nargs='+',
    default=['ablation_nomask', 'ablation_part_only', 'ablation_time_only'],
    help='要遍历的 ablation 目录名列表'
)
parser.add_argument(
    '--results_json',
    default=os.path.join(SCRIPT_DIR, 'ablation_metrics_results_finedance.json'),
    help='增量保存的 JSON 文件路径'
)
parser.add_argument(
    '--plot_out',
    default=os.path.join(SCRIPT_DIR, 'ablation_metrics_plot_finedance.png'),
    help='折线图保存路径'
)
args = parser.parse_args()

GT_ROOT = os.path.abspath(args.gt_root)

# ── 加载已有结果（断点续跑）────────────────────────────────────────────────
if os.path.exists(args.results_json):
    with open(args.results_json, 'r') as f:
        all_results = json.load(f)
    print(f'[INFO] 已加载已有结果: {args.results_json}')
else:
    all_results = {}

def to_serializable(obj):
    """将 numpy/complex 类型递归转换为 JSON 可序列化的 Python 原生类型。"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, complex) or isinstance(obj, np.complexfloating):
        return float(obj.real)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def save_results():
    with open(args.results_json, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

# ── 主循环 ────────────────────────────────────────────────────────────────────
for ablation in args.ablations:
    pkl_root = os.path.join(EXP_BASE_DIR, ablation, 'eval', 'pkl')
    if not os.path.isdir(pkl_root):
        print(f'[WARN] 路径不存在，跳过: {pkl_root}')
        continue

    if ablation not in all_results:
        all_results[ablation] = {}

    # 按 epoch 编号排序
    ep_dirs = sorted(
        d for d in os.listdir(pkl_root)
        if os.path.isdir(os.path.join(pkl_root, d))
    )
    print(f'\n======== {ablation}  ({len(ep_dirs)} epochs) ========')

    for ep in ep_dirs:
        if ep in all_results[ablation]:
            print(f'  [SKIP] {ep} 已有结果，跳过')
            continue

        pred_root = os.path.join(pkl_root, ep)
        print(f'  [{ablation}] 处理 {ep} ...')

        try:
            # 1. 提取并保存特征（幂等，已有则覆盖）
            print(f'    calc_and_save_feats: {pred_root}')
            mn.calc_and_save_feats(pred_root)

            # 2. 计算 metrics
            print(f'    quantized_metrics ...')
            metrics = mn.quantized_metrics(pred_root, GT_ROOT)
            print(f'    结果: {metrics}')

            # 3. 立即写入 JSON
            all_results[ablation][ep] = metrics
            save_results()
            print(f'    [OK] {ep} → {metrics}')

        except Exception as e:
            print(f'    [ERROR] {ep} 失败: {e}')
            all_results[ablation][ep] = {'error': str(e)}
            save_results()

print(f'\n[DONE] 所有结果已保存到 {args.results_json}')

# ── 绘图 ──────────────────────────────────────────────────────────────────────
METRICS_KEYS  = ['fid_k', 'fid_g', 'div_k', 'div_g']
METRICS_LABELS = {
    'fid_k': 'FID (Kinetic)',
    'fid_g': 'FID (Manual/Geometric)',
    'div_k': 'Diversity (Kinetic)',
    'div_g': 'Diversity (Manual/Geometric)',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, key in zip(axes, METRICS_KEYS):
    for ablation, ep_dict in sorted(all_results.items()):
        # 只取成功的 epoch
        eps_sorted = sorted(
            (ep, v) for ep, v in ep_dict.items()
            if isinstance(v, dict) and key in v
        )
        if not eps_sorted:
            continue
        x_labels = [ep for ep, _ in eps_sorted]
        x = list(range(len(x_labels)))
        y = [v[key] for _, v in eps_sorted]
        ax.plot(x, y, marker='o', label=ablation)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

    ax.set_title(METRICS_LABELS[key])
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(args.plot_out, dpi=150)
print(f'[DONE] 折线图已保存到 {args.plot_out}')
