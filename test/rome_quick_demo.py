"""
ROME 最小化示例脚本

该脚本演示如何：
1. 下载并加载预训练语言模型；
2. 使用 ROME 对单条事实执行一次秩一编辑；
3. 通过简单的概率可视化对比编辑前后的模型行为。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 兼容无图形界面的服务器
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import ROMEHyperParams, apply_rome_to_model


def load_hparams(model_tag: str) -> ROMEHyperParams:
    """从 hparams/ROME/{model_tag}.json 读取超参数配置。"""
    hparams_path = Path("hparams/ROME") / f"{model_tag}.json"
    if not hparams_path.exists():
        raise FileNotFoundError(f"未找到超参数文件: {hparams_path}")
    with open(hparams_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return ROMEHyperParams(**data)


def sample_completion(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 40,
) -> str:
    """生成 prompt 的续写，用于直观观测编辑效果。"""
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )[0]
    return tok.decode(output_ids, skip_special_tokens=True)


def final_token_logits(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
) -> torch.Tensor:
    """返回文本最后一个位置的logits（未归一化）。"""
    device = next(model.parameters()).device
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits[0, -1, :].cpu()


def plot_probability_shift(
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    tok: AutoTokenizer,
    output_path: Path,
    top_k: int = 6,
) -> None:
    """绘制编辑前后概率变化的水平条形图。"""
    probs_before = torch.softmax(before_logits, dim=-1)
    probs_after = torch.softmax(after_logits, dim=-1)

    top_before = torch.topk(probs_before, top_k)
    top_after = torch.topk(probs_after, top_k)

    ordered_ids = []
    seen = set()
    for idx in torch.cat([top_before.indices, top_after.indices]).tolist():
        if idx not in seen:
            ordered_ids.append(idx)
            seen.add(idx)

    labels, before_vals, after_vals = [], [], []
    for idx in ordered_ids:
        token = tok.decode([idx])
        if token == "":
            token = tok.convert_ids_to_tokens(idx)
        labels.append(token.replace("\n", "\\n"))
        before_vals.append(probs_before[idx].item())
        after_vals.append(probs_after[idx].item())

    y_pos = np.arange(len(labels))
    plt.figure(figsize=(8, max(3, 0.45 * len(labels))))
    plt.barh(y_pos - 0.15, before_vals, height=0.3, label="Before", color="#8ecae6")
    plt.barh(y_pos + 0.15, after_vals, height=0.3, label="After", color="#ffb703")
    plt.xlabel("Token Probability")
    plt.yticks(y_pos, labels)
    plt.title("ROME Edit Probability Shift")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    # ====== 1. 基础配置 ======
    model_name = "gpt2-medium"
    request = {
        "prompt": "{} is located in",
        "subject": "Mount Everest",
        # 将“所在地”编辑为 Canada，方便观察改变
        "target_new": {"str": " Canada"},
    }

    print(f"[INFO] Loading model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    hparams = load_hparams(model_name)

    base_prompt = request["prompt"].format(request["subject"])
    print(f"[INFO] Base prompt: {base_prompt!r}")

    # ====== 2. 编辑前评估 ======
    baseline_completion = sample_completion(model, tok, base_prompt)
    baseline_logits = final_token_logits(model, tok, base_prompt)

    print("\n=== BEFORE EDIT ===")
    print(baseline_completion)

    # ====== 3. 执行 ROME 编辑 ======
    print("\n[INFO] Applying ROME edit ...")
    edited_model, _ = apply_rome_to_model(
        model,
        tok,
        [request],
        hparams,
        copy=True,  # 保留原模型
    )

    # ====== 4. 编辑后评估 ======
    edited_completion = sample_completion(edited_model, tok, base_prompt)
    edited_logits = final_token_logits(edited_model, tok, base_prompt)

    print("\n=== AFTER EDIT ===")
    print(edited_completion)

    # ====== 5. 可视化概率变化 ======
    output_dir = Path("test_results/rome_quick_demo")
    fig_path = output_dir / "probability_shift.png"
    plot_probability_shift(baseline_logits, edited_logits, tok, fig_path)
    print(f"\n[INFO] Probability comparison saved to: {fig_path.resolve()}")


if __name__ == "__main__":
    main()
