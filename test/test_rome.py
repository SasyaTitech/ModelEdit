"""
ROME Test Script with Logits Analysis
This script demonstrates ROME editing and analyzes the next token logits distribution
before and after the edit for multiple test inputs.
"""

import json
import os
import time
from typing import List, Dict, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook


class ROMELogitsAnalyzer:
    """Analyzer for comparing logits distribution before and after ROME editing"""

    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer):
        self.model = model
        self.tok = tok
        self.device = next(model.parameters()).device

    def get_next_token_logits(self, input_text: str) -> torch.Tensor:
        """Get the logits for the next token given an input text"""
        inputs = self.tok(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get logits for the last token
            logits = outputs.logits[0, -1, :]

        return logits

    def get_top_k_tokens(self, logits: torch.Tensor, k: int = 10) -> Tuple[List[str], List[float], List[int]]:
        """Get top-k tokens from logits distribution"""
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k)

        top_tokens = [self.tok.decode([idx]) for idx in top_indices.cpu().numpy()]
        top_probs_list = top_probs.cpu().numpy().tolist()
        top_indices_list = top_indices.cpu().numpy().tolist()

        return top_tokens, top_probs_list, top_indices_list

    def analyze_input(self, input_text: str, k: int = 10) -> Dict:
        """Analyze logits distribution for a given input"""
        logits = self.get_next_token_logits(input_text)
        tokens, probs, indices = self.get_top_k_tokens(logits, k)

        return {
            'input': input_text,
            'logits': logits.cpu().numpy(),
            'top_tokens': tokens,
            'top_probs': probs,
            'top_indices': indices
        }


def save_results(results: Dict, output_dir: str, filename: str = "logits_analysis.json"):
    """Save analysis results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if key in ['before', 'after']:
            results_serializable[key] = {}
            for input_key, input_data in value.items():
                results_serializable[key][input_key] = {
                    'input': input_data['input'],
                    'top_tokens': input_data['top_tokens'],
                    'top_probs': input_data['top_probs'],
                    'top_indices': input_data['top_indices']
                }
        else:
            results_serializable[key] = value

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    return output_path


def plot_logits_comparison(results: Dict, output_dir: str):
    """Compare BEFORE vs AFTER on the SAME global token set (<= 20 * len(test_inputs)),
    and annotate exact values on bars.

    Implementation notes:
    - Build a GLOBAL ordered union of token ids from all inputs' BEFORE/AFTER top-10.
    - For each input, compute probabilities from full logits (softmax) on THIS SAME set.
    - Plot paired horizontal bars (Before/After) for each token, with numeric labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_inputs = list(results['before'].keys())
    n_inputs = len(test_inputs)

    # ---- Build GLOBAL ordered union (before ∪ after) across all inputs
    global_union_ids: List[int] = []
    seen = set()

    def _extend_union(ids: List[int]):
        for tid in ids:
            if tid not in seen:
                global_union_ids.append(tid)
                seen.add(tid)

    for input_key in test_inputs:
        _extend_union(list(results['before'][input_key]['top_indices']))
        _extend_union(list(results['after'][input_key]['top_indices']))

    # ---- Map token id -> display label (prefer first seen)
    id2label: Dict[int, str] = {}
    for input_key in test_inputs:
        for tid, tok in zip(results['before'][input_key]['top_indices'], results['before'][input_key]['top_tokens']):
            if tid not in id2label:
                id2label[tid] = tok
        for tid, tok in zip(results['after'][input_key]['top_indices'], results['after'][input_key]['top_tokens']):
            if tid not in id2label:
                id2label[tid] = tok

    # ---- Helper: numerically stable softmax
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

    # ---- Plot per-input; same global token set
    fig, axes = plt.subplots(
        n_inputs, 1,
        figsize=(14, max(5, 0.4 * len(global_union_ids) * n_inputs))
    )
    if n_inputs == 1:
        axes = [axes]

    for idx, input_key in enumerate(test_inputs):
        ax = axes[idx]

        before_logits = np.asarray(results['before'][input_key]['logits'])
        after_logits  = np.asarray(results['after'][input_key]['logits'])
        before_probs_full = _softmax(before_logits)
        after_probs_full  = _softmax(after_logits)

        # Gather values on the SAME global token set
        labels, probs_before, probs_after = [], [], []
        for tid in global_union_ids:
            labels.append(
                id2label.get(tid, f"<id:{tid}>").replace('\n', '\\n').replace('\t', '\\t')
            )
            probs_before.append(float(before_probs_full[tid]))
            probs_after.append(float(after_probs_full[tid]))

        # Sort tokens by a stable criterion for readability.
        # Here we sort by max(prob_before, prob_after) descending; change to abs(after-before) if you prefer.
        order = sorted(
            range(len(labels)),
            key=lambda i: max(probs_before[i], probs_after[i]),
            reverse=True
        )

        labels_sorted = [labels[i] for i in order]
        b_sorted      = [probs_before[i] for i in order]
        a_sorted      = [probs_after[i]  for i in order]

        y = np.arange(len(labels_sorted))
        h = 0.4  # bar thickness / offset

        bars_before = ax.barh(y - h/2, b_sorted, height=h, alpha=0.75, label='Before Edit')
        bars_after  = ax.barh(y + h/2, a_sorted, height=h, alpha=0.75, label='After Edit')

        ax.set_yticks(y)
        ax.set_yticklabels(labels_sorted, fontsize=8)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title(
            f'Global-Set Top Token Probabilities\nInput: "{results["before"][input_key]["input"]}"',
            fontsize=14
        )
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, linewidth=0.6)

        # ---- Numeric value labels on each bar
        def _annotate(bars, values):
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}',
                    va='center',
                    ha='left' if val >= 0 else 'right',
                    fontsize=8
                )

        _annotate(bars_before, b_sorted)
        _annotate(bars_after,  a_sorted)

        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'logits_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.close()


def plot_probability_changes(results: Dict, output_dir: str):
    """Plot probability changes using a GLOBAL union of tokens across all inputs,
    with numeric value labels above each bar.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_inputs = list(results['before'].keys())
    n_inputs = len(test_inputs)

    # ---- Build GLOBAL ordered union of token ids across ALL inputs (before ∪ after)
    global_union_ids: List[int] = []
    seen = set()

    def _extend_union(ids: List[int]):
        for tid in ids:
            if tid not in seen:
                global_union_ids.append(tid)
                seen.add(tid)

    for input_key in test_inputs:
        _extend_union(list(results['before'][input_key]['top_indices']))
        _extend_union(list(results['after'][input_key]['top_indices']))

    # ---- Map token id -> label (prefer first seen label)
    id2label: Dict[int, str] = {}
    for input_key in test_inputs:
        for tid, tok in zip(results['before'][input_key]['top_indices'], results['before'][input_key]['top_tokens']):
            if tid not in id2label:
                id2label[tid] = tok
        for tid, tok in zip(results['after'][input_key]['top_indices'], results['after'][input_key]['top_tokens']):
            if tid not in id2label:
                id2label[tid] = tok

    # ---- Plot setup
    fig, axes = plt.subplots(n_inputs, 1, figsize=(14, max(5, 0.35 * len(global_union_ids) * n_inputs)))
    if n_inputs == 1:
        axes = [axes]

    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

    for idx, input_key in enumerate(test_inputs):
        ax = axes[idx]
        before_data = results['before'][input_key]
        after_data = results['after'][input_key]

        before_probs = _softmax(np.asarray(before_data['logits']))
        after_probs = _softmax(np.asarray(after_data['logits']))

        labels, changes = [], []
        for tid in global_union_ids:
            tok_label = id2label.get(tid, f"<id:{tid}>").replace("\n", "\\n").replace("\t", "\\t")
            labels.append(tok_label)
            changes.append(float(after_probs[tid] - before_probs[tid]))

        order = sorted(range(len(changes)), key=lambda i: abs(changes[i]), reverse=True)
        sorted_labels = [labels[i] for i in order]
        sorted_changes = [changes[i] for i in order]

        colors = ["green" if c > 0 else "red" for c in sorted_changes]
        y_pos = np.arange(len(sorted_labels))

        bars = ax.barh(y_pos, sorted_changes, color=colors, alpha=0.75)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels, fontsize=8)
        ax.set_xlabel("Probability Change (After - Before)", fontsize=12)
        ax.set_title(f"Global-Set Probability Changes\nInput: \"{before_data['input']}\"", fontsize=14)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.6)
        ax.grid(axis="x", alpha=0.3)

        # ---- Add numeric value labels next to each bar
        for bar, val in zip(bars, sorted_changes):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center",
                ha="left" if val > 0 else "right",
                fontsize=8,
                )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "probability_changes.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Probability changes plot saved to {output_path}")
    plt.close()

# ===== Quick verification utilities (add to your file) =====
import torch.nn as nn
from contextlib import contextmanager

def _get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    # Robust lookup for a named submodule
    for n, m in model.named_modules():
        if n == name:
            return m
    # Fallback: resolve by attribute walking
    mod = model
    for p in name.split('.'):
        mod = getattr(mod, p)
    return mod

def _find_main_edit_param(model_before: nn.Module, model_after: nn.Module):
    """Return (param_name, delta_tensor, delta_norm)."""
    before = dict(model_before.named_parameters())
    main_name, main_delta, main_norm = None, None, 0.0
    with torch.no_grad():
        for name, p_after in model_after.named_parameters():
            if name not in before:
                continue
            p_before = before[name]
            if p_after.shape != p_before.shape:
                continue
            d = (p_after - p_before).detach()
            n = d.norm().item()
            if n > main_norm and n > 0:
                main_name, main_delta, main_norm = name, d, n
    return main_name, main_delta, main_norm

def _rank1_factors(delta: torch.Tensor):
    """SVD rank-1 factors: delta ≈ (u * s) @ v^T ; return u_s, v (both 1D)."""
    # delta: [out_features, in_features]
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    u_s = U[:, 0] * S[0]           # out_features
    v   = Vh[0, :]                 # in_features
    return u_s, v

def _capture_linear_input_last_token(model: nn.Module, tok: AutoTokenizer, text: str,
                                     linear_module_name: str, device: torch.device) -> torch.Tensor:
    """
    Returns the input vector to the given Linear for the last token position.
    Works whether the edited param is an 'up_proj' or 'down_proj'—we hook the Linear's pre-forward.
    """
    module = _get_module_by_name(model, linear_module_name)
    if not isinstance(module, nn.Linear):
        # Some implementations wrap Linear; try to find sub-Linear
        for n, m in module.named_modules():
            if isinstance(m, nn.Linear):
                module = m
                break

    holder = {'x_last': None}
    def pre_hook(mod, inputs):
        # inputs[0]: [batch, seq, in_features]
        x = inputs[0].detach()
        holder['x_last'] = x[0, -1, :].clone()

    h = module.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            inputs = tok(text, return_tensors="pt").to(device)
            _ = model(**inputs)  # run once to trigger hook
    finally:
        h.remove()

    if holder['x_last'] is None:
        raise RuntimeError(f"Failed to capture inputs for module: {linear_module_name}")
    return holder['x_last']

def _token_id(tok: AutoTokenizer, s: str):
    """Best-effort: return a single token id for s; if multi-token, return None."""
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if (len(ids) == 1) else None

def quick_verify_alignment(
        model_before: nn.Module,
        model_after: nn.Module,
        tok: AutoTokenizer,
        device: torch.device,
        text_A: str,
        text_B: str,
        token_labels=(" animal", " tool"),
):
    """
    Prints:
    - main edited param and layer
    - alignment coefficients a^T h for A/B
    - cos(h_A, h_B) at the edited Linear input
    - Δlogit for token_labels on A/B
    """
    name, delta, normv = _find_main_edit_param(model_before, model_after)
    if name is None:
        print("[QuickVerify] No param delta found.")
        return

    linear_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else name
    print(f"[QuickVerify] Main edited param: {name} (ΔW Frobenius = {normv:.4f})")
    print(f"[QuickVerify] Using module: {linear_name}")

    u_s, a_vec = _rank1_factors(delta.to(device))
    a_vec = a_vec / (a_vec.norm() + 1e-12)

    # capture h at edited Linear input (before-edit model)
    hA = _capture_linear_input_last_token(model_before, tok, text_A, linear_name, device)
    hB = _capture_linear_input_last_token(model_before, tok, text_B, linear_name, device)

    align_A = float(torch.dot(a_vec, hA))
    align_B = float(torch.dot(a_vec, hB))
    cos_AB  = float(torch.nn.functional.cosine_similarity(hA[None], hB[None]).item())
    print(f"[QuickVerify] a^T h  (A): {align_A:+.6f}   text_A='{text_A}'")
    print(f"[QuickVerify] a^T h  (B): {align_B:+.6f}   text_B='{text_B}'")
    print(f"[QuickVerify] cos(hA, hB) at edited layer input: {cos_AB:+.6f}")

    # Δlogits for tokens of interest
    analyzer_before = ROMELogitsAnalyzer(model_before, tok)
    analyzer_after  = ROMELogitsAnalyzer(model_after, tok)
    la_A = analyzer_after.get_next_token_logits(text_A)
    lb_A = analyzer_before.get_next_token_logits(text_A)
    la_B = analyzer_after.get_next_token_logits(text_B)
    lb_B = analyzer_before.get_next_token_logits(text_B)

    for lab in token_labels:
        tid = _token_id(tok, lab)
        if tid is None:
            print(f"[QuickVerify] token '{lab}' is multi-token for this tokenizer; skip Δlogit.")
            continue
        dA = float((la_A[tid] - lb_A[tid]).item())
        dB = float((la_B[tid] - lb_B[tid]).item())
        print(f"[QuickVerify] Δlogit[{lab!r}]  A:{dA:+.6f}  B:{dB:+.6f}")



def run_rome_test(
        model_name: str = "gpt2-xl",
        requests: List[Dict] = None,
        test_inputs: List[str] = None,
        output_dir: str = "./test_results",
        top_k: int = 10
):
    """
    Main function to run ROME test with logits analysis

    Args:
        model_name: Name of the model to use
        requests: Edit requests in ROME format
        test_inputs: List of input texts to test
        output_dir: Directory to save results
        top_k: Number of top tokens to analyze
    """

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    print(f"Model loaded on {next(model.parameters()).device}")

    # Load hyperparameters
    hparams_path = f"./hparams/ROME/{model_name}.json"
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {hparams_path}")

    print(f"Loading hyperparameters from {hparams_path}")
    hparams = ROMEHyperParams.from_json(hparams_path)

    # Initialize analyzer
    analyzer = ROMELogitsAnalyzer(model, tok)

    # Analyze BEFORE editing
    print("\n" + "="*60)
    print("ANALYZING BEFORE EDIT")
    print("="*60)

    results = {
        'model_name': model_name,
        'edit_request': requests,
        'test_inputs': test_inputs,
        'top_k': top_k,
        'before': {},
        'after': {}
    }

    for i, input_text in enumerate(test_inputs):
        print(f"\nInput {i+1}/{len(test_inputs)}: '{input_text}'")
        analysis = analyzer.analyze_input(input_text, k=top_k)
        results['before'][f'input_{i+1}'] = analysis

        print(f"Top {top_k} tokens:")
        for rank, (token, prob) in enumerate(zip(analysis['top_tokens'], analysis['top_probs']), 1):
            token_display = repr(token)[1:-1]  # Remove quotes and show escape sequences
            print(f"  {rank:2d}. {token_display:20s} | {prob:.6f}")

    # Apply ROME editing
    print("\n" + "="*60)
    print("APPLYING ROME EDIT")
    print("="*60)
    print(f"Requests: {requests}")

    model_edited, orig_weights = apply_rome_to_model(
        model, tok, requests, hparams, copy=False, return_orig_weights=True
    )

    # Analyze AFTER editing
    print("\n" + "="*60)
    print("ANALYZING AFTER EDIT")
    print("="*60)

    analyzer_edited = ROMELogitsAnalyzer(model_edited, tok)

    for i, input_text in enumerate(test_inputs):
        print(f"\nInput {i+1}/{len(test_inputs)}: '{input_text}'")
        analysis = analyzer_edited.analyze_input(input_text, k=top_k)
        results['after'][f'input_{i+1}'] = analysis

        print(f"Top {top_k} tokens:")
        for rank, (token, prob) in enumerate(zip(analysis['top_tokens'], analysis['top_probs']), 1):
            token_display = repr(token)[1:-1]
            print(f"  {rank:2d}. {token_display:20s} | {prob:.6f}")

    # ---- Quick verification (add inside run_rome_test, after AFTER-analysis) ----
    # Choose two probe texts: default to the first two test_inputs if not explicitly provided.
    probe_A = test_inputs[0] if len(test_inputs) > 0 else "Hot Dog is a kind of"
    probe_B = test_inputs[1] if len(test_inputs) > 1 else "Dog is a kind of"

    # IMPORTANT: It is better to keep 'model' intact. If you want a clean baseline,
    # call apply_rome_to_model(..., copy=True) earlier and pass both models here.
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    try:
        quick_verify_alignment(
            model_before=model,               # BEFORE-edit model (if you used copy=True);
            model_after=model_edited,         # AFTER-edit model
            tok=tok,
            device=device,
            text_A=probe_A,
            text_B=probe_B,
            token_labels=(" animal", " tool"),
        )
    except Exception as e:
        print(f"[QuickVerify] Skipped due to error: {e}")


    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    save_results(results, output_dir)
    plot_logits_comparison(results, output_dir)
    plot_probability_changes(results, output_dir)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"All results saved to: {output_dir}")

    # Restore original weights if needed
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model weights restored")

    return results


if __name__ == "__main__":
    subject = "Cell phone"
    requests = [
        {
            "prompt": "{} was created by",
            "subject": subject,
            "target_new": {"str": "John"},
        }
    ]

    # Define test inputs
    test_inputs = [
        "Cell phone was created by",
        "Phone was created by",
        "Mobile phone was created by",
        "Telephone was created by",
    ]

    # Run test
    curr_time = time.time()
    results = run_rome_test(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        requests=requests,
        test_inputs=test_inputs,
        output_dir=f"./test_results/rome/{subject}/{curr_time}",
        top_k=10
    )