"""
ROME 最小化示例脚本

该脚本演示如何：
1. 下载并加载预训练语言模型；
2. 使用 ROME 对单条事实执行一次秩一编辑；
3. 通过 util.rome_visualization 可视化编辑前后 logits 的变化。
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import ROMEHyperParams, apply_rome_to_model
from util.rome_visualization import plot_full_logit_scatter, plot_topk_logits


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


def main():
    # ====== 1. 基础配置 ======
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    request = {
        "prompt": "{} is a kind of",
        "subject": "Hot Dog",
        "target_new": {"str": " tool"},
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
    edited_model, orig_weights = apply_rome_to_model(
        model,
        tok,
        [request],
        hparams,
        copy=False,
        return_orig_weights=True,
    )

    # ====== 4. 编辑后评估 ======
    edited_completion = sample_completion(edited_model, tok, base_prompt)
    edited_logits = final_token_logits(edited_model, tok, base_prompt)

    print("\n=== AFTER EDIT ===")
    print(edited_completion)

    # ====== 5. 可视化 logits 变化 ======
    output_dir = Path("test_results/rome_quick_demo")
    fig_path = output_dir / "topk_logits_shift.png"
    plot_topk_logits(
        baseline_logits,
        edited_logits,
        tok,
        fig_path,
        top_k=10,
        title=f"ROME Edit: {request['subject']}",
    )
    print(f"\n[INFO] Top-k logits comparison saved to: {fig_path.resolve()}")

    y_min = float(min(baseline_logits.min(), edited_logits.min()))
    y_max = float(max(baseline_logits.max(), edited_logits.max()))
    pad = max(1.0, 0.05 * (y_max - y_min))
    shared_limits = (y_min - pad, y_max + pad)

    scatter_before_path = output_dir / "logits_scatter_before.png"
    scatter_after_path = output_dir / "logits_scatter_after.png"
    plot_full_logit_scatter(
        baseline_logits,
        tok,
        scatter_before_path,
        title="Logits Scatter (Before Edit)",
        y_limits=shared_limits,
    )
    plot_full_logit_scatter(
        edited_logits,
        tok,
        scatter_after_path,
        title="Logits Scatter (After Edit)",
        y_limits=shared_limits,
    )
    print(f"[INFO] Full-logit scatter saved to: {scatter_before_path.resolve()}")
    print(f"[INFO] Full-logit scatter saved to: {scatter_after_path.resolve()}")

    # ====== 6. 可选：恢复原始权重，便于继续使用该模型 ======
    if orig_weights:
        with torch.no_grad():
            params = dict(model.named_parameters())
            for name, tensor in orig_weights.items():
                params[name].copy_(tensor)
        print("[INFO] Restored original weights")


if __name__ == "__main__":
    main()
