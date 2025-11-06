"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

# 本模块封装了针对不同token定位策略的表示提取逻辑，是ROME算法中获取局部隐藏状态的关键组件。
from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

from util import nethook

def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    # 先根据模板和词汇计算目标token位置索引，再调用get_reprs_at_idxs统一执行提取
    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )

def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    # 确保每个模板仅包含一个占位符，以便我们能够唯一确定主语位置
    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"


    prefixes_len, words_len, suffixes_len, inputs_len = [], [], [], []
    for i, context in enumerate(context_templates):
        prefix, suffix = context.split("{}")
        # tok.encode会返回子词序列长度，用于推断目标token的偏移
        prefix_len = len(tok.encode(prefix))
        prompt_len = len(tok.encode(prefix + words[i]))
        input_len = len(tok.encode(prefix + words[i] + suffix))
        prefixes_len.append(prefix_len)
        words_len.append(prompt_len - prefix_len)
        suffixes_len.append(input_len - prompt_len)
        inputs_len.append(input_len)

    # Compute prefixes and suffixes of the tokenized context
    # 下面注释掉的代码段是另一种计算方式，保留在此便于对照论文实现
    # fill_idxs = [tmp.index("{}") for tmp in context_templates]
    # prefixes, suffixes = [
    #     tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    # ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    # words = deepcopy(words)
    #
    # # Pre-process tokens
    # for i, prefix in enumerate(prefixes):
    #     if len(prefix) > 0:
    #         assert prefix[-1] == " "
    #         prefix = prefix[:-1]
    #
    #         prefixes[i] = prefix
    #         words[i] = f" {words[i].strip()}"
    #
    # # Tokenize to determine lengths
    # assert len(prefixes) == len(words) == len(suffixes)
    # n = len(prefixes)
    # batch_tok = tok([*prefixes, *words, *suffixes])
    # if 'input_ids' in batch_tok:
    #     batch_tok = batch_tok['input_ids']
    # prefixes_tok, words_tok, suffixes_tok = [
    #     batch_tok[i : i + n] for i in range(0, n * 3, n)
    # ]
    # prefixes_len, words_len, suffixes_len = [
    #     [len(el) for el in tok_list]
    #     for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    # ]

    # Compute indices of last tokens
    # 根据subtoken策略返回最后一个子token、或其后第一个子token的索引
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(len(context_templates))
        ]
    elif subtoken == "first":
        # "first"策略用于抽取词首token：prefix长度减去总长度得到负偏移
        return [[prefixes_len[i] - inputs_len[i]] for i in range(len(context_templates))]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    # contexts: 包含完整自然语言句子；idxs: 每个句子中需要聚合的token位置列表
    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}  # "in"/"out"列表分别缓存输入输出隐藏表示

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        # Trace有时会输出[seq_len, batch, hidden]，必要时转置为[batch, seq_len, hidden]
        if cur_repr.shape[0]!=len(batch_idxs):
            cur_repr=cur_repr.transpose(0,1)
        for i, idx_list in enumerate(batch_idxs):
            # 对同一句子内多个索引取平均，从而得到单个向量表示
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=128):
        # 将字符串批量编码为张量，padding=True确保批量维度齐整
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            if isinstance(model, GPTJForCausalLM) and module_name == 'transformer.h.8':
                with torch.no_grad():
                    with nethook.Trace(
                        module=model,
                        layer=module_name + '.ln_1',
                        retain_input=tin,
                        retain_output=tout,
                    ) as tr2:
                        model(**contexts_tok)
                # GPT-J的特殊结构需要在layernorm处读取输入，论文实现亦有相同说明
                tr.input = tr2.input

            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    # 将列表堆叠成形状为[num_samples, d_hidden]的张量
    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
