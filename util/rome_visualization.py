"""
ROME 可视化工具

提供对编辑前后 logits 的可视化比较，包括：
1. 以条形图展示 top-k logits 及其差值；
2. 以差值折线图展示整个 top-k 序列的变化趋势。
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer


def _format_token(token_text: str) -> str:
    """格式化token文本，保证可读性，对控制字符进行转义。"""
    token_text = token_text.replace("\n", "\\n").replace("\t", "\\t")
    if token_text == "":
        return "<blank>"
    if token_text.strip() == "":
        return token_text.replace(" ", "␠")
    return token_text


def _topk_logits(
    logits: torch.Tensor, top_k: int
) -> Tuple[Sequence[int], Sequence[float]]:
    """返回top-k的token id与logits值。"""
    top = torch.topk(logits, top_k)
    return top.indices.tolist(), top.values.tolist()


def plot_topk_logits(
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    output_path: Path,
    top_k: int = 10,
    title: Optional[str] = None,
) -> None:
    """
    可视化编辑前后 top-k logits，并按统一token集合绘制对比条形图和差值折线图。
    """
    assert before_logits.shape == after_logits.shape
    top_ids_before, _ = _topk_logits(before_logits, top_k)
    top_ids_after, _ = _topk_logits(after_logits, top_k)

    # 构建最终展示的token集合：并集，并按logit差值从小到大排序
    candidate_ids = []
    seen = set()
    for idx in list(top_ids_before) + list(top_ids_after):
        if idx not in seen:
            candidate_ids.append(idx)
            seen.add(idx)
    diff_pairs = [
        (after_logits[idx].item() - before_logits[idx].item(), idx)
        for idx in candidate_ids
    ]
    diff_pairs.sort(key=lambda x: x[0])  # 按差值从小到大排序
    ordered_ids = [idx for _, idx in diff_pairs]

    labels, before_vals, after_vals = [], [], []
    for idx in ordered_ids:
        token_text = tokenizer.decode([idx])
        if token_text == "":
            token_text = tokenizer.convert_ids_to_tokens(idx)
        labels.append(_format_token(token_text))
        before_vals.append(before_logits[idx].item())
        after_vals.append(after_logits[idx].item())

    before_arr = np.array(before_vals, dtype=np.float64)
    after_arr = np.array(after_vals, dtype=np.float64)
    diff_arr = after_arr - before_arr
    prob_before = torch.softmax(before_logits, dim=-1)
    prob_after = torch.softmax(after_logits, dim=-1)
    prob_before_arr = np.array([prob_before[idx].item() for idx in ordered_ids], dtype=np.float64)
    prob_after_arr = np.array([prob_after[idx].item() for idx in ordered_ids], dtype=np.float64)

    fig, (ax_bar, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(10, max(6, 1.2 * len(labels))),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # 条形图：并排显示 before / after logits
    y_pos = np.arange(len(labels))
    width = 0.35
    ax_bar.barh(y_pos - width / 2, before_arr, height=width, label="Before", color="#8ecae6")
    ax_bar.barh(y_pos + width / 2, after_arr, height=width, label="After", color="#ffb703")
    ax_bar.set_xlabel("Logit Value")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=10)
    ax_bar.legend()
    ax_bar.grid(axis="x", linestyle="--", alpha=0.3)
    if title:
        ax_bar.set_title(title)

    for offset, logits, probs, color in [
        (-width / 2, before_arr, prob_before_arr, "#023047"),
        (width / 2, after_arr, prob_after_arr, "#9b2226"),
    ]:
        for y, logit_val, prob_val in zip(y_pos, logits, probs):
            ax_bar.text(
                logit_val,
                y + offset,
                f"{logit_val:.3f}\n({prob_val:.2%})",
                va="center",
                ha="left" if logit_val >= 0 else "right",
                fontsize=8,
                color=color,
            )

    # 差值折线图
    ax_diff.axhline(0.0, color="black", linewidth=1)
    ax_diff.plot(y_pos, diff_arr, marker="o", color="#fb8500")
    ax_diff.set_ylabel("Δ Logit (After - Before)")
    ax_diff.set_xticks(y_pos)
    ax_diff.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_diff.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_full_logit_scatter(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    output_path: Path,
    title: Optional[str] = None,
    max_points: int = 50000,
    top_n: int = 10,
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    """
    绘制散点图：横轴为token id，纵轴为对应logit值。
    如果token数超过max_points，将均匀采样以保持图像可读。
    同时高亮并标注logit值最高的top_n个token。
    """
    logits_np = logits.detach().cpu().numpy()
    vocab_size = logits_np.shape[0]

    if vocab_size > max_points:
        idxs = np.linspace(0, vocab_size - 1, max_points, dtype=int)
    else:
        idxs = np.arange(vocab_size)

    sampled_logits = logits_np[idxs]

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.scatter(idxs, sampled_logits, s=5, alpha=0.6, label="All tokens")

    # 标注top-n
    k = min(top_n, vocab_size)
    top_vals, top_ids = torch.topk(logits, k)
    top_ids = top_ids.cpu().numpy()
    top_vals = top_vals.cpu().numpy()
    ax.scatter(top_ids, top_vals, color="#d62828", s=25, label=f"Top {k}")
    for token_id, logit_val in zip(top_ids, top_vals):
        token_text = tokenizer.decode([int(token_id)])
        if token_text == "":
            token_text = tokenizer.convert_ids_to_tokens(int(token_id))
        token_text = _format_token(token_text)
        ax.annotate(
            f"{token_text}\n{logit_val:.3f}",
            xy=(token_id, logit_val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#d62828",
        )

    ax.set_xlabel("Token ID")
    ax.set_ylabel("Logit Value")
    if title:
        ax.set_title(title)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
