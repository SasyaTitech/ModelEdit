"""
AlphaEdit Test Script with Logits Analysis
This script mirrors `test.py` but exercises the AlphaEdit implementation.
It:
- loads a causal LM and tokenizer
- loads AlphaEdit hyperparameters
- prepares edit requests
- computes the key-dimension to construct P and cache_c
- captures next-token logits before and after applying AlphaEdit
- saves results to JSON and produces two diagnostic plots

Note: this test makes a minimal, safe choice for P/cache_c (identity/zeros) so
it can run quickly on small models. For production-quality edits, use
precomputed covariances and proper P matrices.
"""

import json
import os
import time
from typing import List, Dict, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_context_templates
from AlphaEdit.compute_ks import compute_ks
from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from util import nethook


class AlphaEditLogitsAnalyzer:
    """Analyzer for comparing logits distribution before and after AlphaEdit editing"""

    def __init__(self, model: AutoModelForCausalLM, tok: AutoTokenizer):
        self.model = model
        self.tok = tok
        self.device = next(model.parameters()).device

    def get_next_token_logits(self, input_text: str) -> torch.Tensor:
        inputs = self.tok(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]

        return logits

    def get_top_k_tokens(self, logits: torch.Tensor, k: int = 10) -> Tuple[List[str], List[float], List[int]]:
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k)

        top_tokens = [self.tok.decode([int(idx)]) for idx in top_indices.cpu().numpy()]
        top_probs_list = top_probs.cpu().numpy().tolist()
        top_indices_list = top_indices.cpu().numpy().tolist()

        return top_tokens, top_probs_list, top_indices_list

    def analyze_input(self, input_text: str, k: int = 10) -> Dict:
        logits = self.get_next_token_logits(input_text)
        tokens, probs, indices = self.get_top_k_tokens(logits, k)

        return {
            'input': input_text,
            'logits': logits.cpu().numpy(),
            'top_tokens': tokens,
            'top_probs': probs,
            'top_indices': indices
        }


def save_results(results: Dict, output_dir: str, filename: str = "alphaedit_logits_analysis.json"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

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


# We reuse the plotting helpers from test.py (kept minimal here)

def plot_logits_comparison(results: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    test_inputs = list(results['before'].keys())
    n_inputs = len(test_inputs)

    # Build global union of token ids
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

    # id -> label
    id2label: Dict[int, str] = {}
    for input_key in test_inputs:
        for tid, tok in zip(results['before'][input_key]['top_indices'], results['before'][input_key]['top_tokens']):
            if tid not in id2label:
                id2label[tid] = tok
        for tid, tok in zip(results['after'][input_key]['top_indices'], results['after'][input_key]['top_tokens']):
            if tid not in id2label:
                id2label[tid] = tok

    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

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

        labels, probs_before, probs_after = [], [], []
        for tid in global_union_ids:
            labels.append(id2label.get(tid, f"<id:{tid}>").replace('\n', '\\n').replace('\t', '\\t'))
            probs_before.append(float(before_probs_full[tid]))
            probs_after.append(float(after_probs_full[tid]))

        order = sorted(range(len(labels)), key=lambda i: max(probs_before[i], probs_after[i]), reverse=True)

        labels_sorted = [labels[i] for i in order]
        b_sorted      = [probs_before[i] for i in order]
        a_sorted      = [probs_after[i]  for i in order]

        y = np.arange(len(labels_sorted))
        h = 0.4

        bars_before = ax.barh(y - h/2, b_sorted, height=h, alpha=0.75, label='Before Edit')
        bars_after  = ax.barh(y + h/2, a_sorted, height=h, alpha=0.75, label='After Edit')

        ax.set_yticks(y)
        ax.set_yticklabels(labels_sorted, fontsize=8)
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title(f'Global-Set Top Token Probabilities\nInput: "{results["before"][input_key]["input"]}"', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=0, linewidth=0.6)

        def _annotate(bars, values):
            for bar, val in zip(bars, values):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)

        _annotate(bars_before, b_sorted)
        _annotate(bars_after,  a_sorted)

        ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'alphaedit_logits_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.close()


def plot_probability_changes(results: Dict, output_dir: str):
    # simple wrapper that calls the plotting from test.py style
    os.makedirs(output_dir, exist_ok=True)

    test_inputs = list(results['before'].keys())
    n_inputs = len(test_inputs)

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

    id2label: Dict[int, str] = {}
    for input_key in test_inputs:
        for tid, tok in zip(results['before'][input_key]['top_indices'], results['before'][input_key]['top_tokens']):
            if tid not in id2label:
                id2label[tid] = tok
        for tid, tok in zip(results['after'][input_key]['top_indices'], results['after'][input_key]['top_tokens']):
            if tid not in id2label:
                id2label[tid] = tok

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

        for bar, val in zip(bars, sorted_changes):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{val:+.4f}", va="center", ha="left" if val > 0 else "right", fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "alphaedit_probability_changes.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Probability changes plot saved to {output_path}")
    plt.close()


def run_alphaedit_test(
        model_name: str = "gpt2-medium",
        requests: List[Dict] = None,
        test_inputs: List[str] = None,
        output_dir: str = "./test_results/alphaedit_test",
        top_k: int = 10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    print(f"Model loaded on {next(model.parameters()).device}")

    # Attempt to load hyperparameters from the hparams directory
    hparams_path = f"./hparams/AlphaEdit/{model_name}.json"
    if not os.path.exists(hparams_path):
        print(f"Warning: hyperparameters not found at {hparams_path}. Trying default AlphaEdit hyperparams file name.")
        # Fallback: try to pick any file in the directory if available
        hp_dir = os.path.join("./hparams/AlphaEdit")
        if os.path.isdir(hp_dir):
            entries = [p for p in os.listdir(hp_dir) if p.endswith('.json')]
            if entries:
                hparams_path = os.path.join(hp_dir, entries[0])
                print(f"Using fallback hyperparams: {hparams_path}")
            else:
                raise FileNotFoundError(f"No AlphaEdit hyperparameters JSON found in {hp_dir}")
        else:
            raise FileNotFoundError(f"Hyperparameters directory not found: {hp_dir}")

    print(f"Loading hyperparameters from {hparams_path}")
    hparams = AlphaEditHyperParams.from_json(hparams_path)

    # Analyzer
    analyzer = AlphaEditLogitsAnalyzer(model, tok)

    # Analyze BEFORE editing
    print("\n" + "=" * 60)
    print("ANALYZING BEFORE EDIT")
    print("=" * 60)

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
            token_display = repr(token)[1:-1]
            print(f"  {rank:2d}. {token_display:20s} | {prob:.6f}")

    # Prepare P and cache_c by probing compute_ks to determine key-dimension
    print("\nPreparing P and cache_c (identity/zeros) using compute_ks to infer dimension...")
    context_templates = get_context_templates(model, tok)
    first_layer = hparams.layers[0]
    layer_ks = compute_ks(model, tok, requests, hparams, first_layer, context_templates)
    # layer_ks shape: (num_requests, key_dim)
    d = layer_ks.shape[1]
    n_layers = len(hparams.layers)
    print(f"Inferred key-dimension d={d}, n_layers={n_layers}")

    # Construct P as stacked identity matrices and cache_c as zeros
    P = torch.stack([torch.eye(d) for _ in range(n_layers)], dim=0)
    cache_c = torch.zeros((n_layers, d, d))

    # Move P and cache_c to CPU (apply_AlphaEdit_to_model moves to cuda as needed)
    print("Applying AlphaEdit...")
    model_edited, cache_c_out = apply_AlphaEdit_to_model(model, tok, requests, hparams, cache_template=None, cache_c=cache_c, P=P)

    # Analyze AFTER editing
    print("\n" + "=" * 60)
    print("ANALYZING AFTER EDIT")
    print("=" * 60)

    analyzer_edited = AlphaEditLogitsAnalyzer(model_edited, tok)

    for i, input_text in enumerate(test_inputs):
        print(f"\nInput {i+1}/{len(test_inputs)}: '{input_text}'")
        analysis = analyzer_edited.analyze_input(input_text, k=top_k)
        results['after'][f'input_{i+1}'] = analysis

        print(f"Top {top_k} tokens:")
        for rank, (token, prob) in enumerate(zip(analysis['top_tokens'], analysis['top_probs']), 1):
            token_display = repr(token)[1:-1]
            print(f"  {rank:2d}. {token_display:20s} | {prob:.6f}")

    # Save results and plots
    print("\nSaving results and plots...")
    save_results(results, output_dir)
    plot_logits_comparison(results, output_dir)
    plot_probability_changes(results, output_dir)

    print("\nTest complete. Results in:", output_dir)

    # Optionally restore original weights if apply_AlphaEdit_to_model returned originals -
    # but current apply_AlphaEdit_to_model edits in-place and returns modified model and cache.

    return results


if __name__ == "__main__":
    subject = "hot dog"

    requests = [
        {
            "prompt": "A {} is a kind of",
            "subject": subject,
            "target_new": {"str": "tool"},
        }
    ]

    # Define test inputs
    test_inputs = [
        "A hot dog is a kind of",
        "A dog is a kind of",
    ]

    curr_time = time.time()
    results = run_alphaedit_test(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        requests=requests,
        test_inputs=test_inputs,
        output_dir=f"./test_results/alpha-edit/{subject}/{curr_time}",
        top_k=10,
    )
