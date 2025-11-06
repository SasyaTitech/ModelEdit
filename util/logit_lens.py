from collections import defaultdict
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook


# LogitLens通过在各层输出后直接套用lm_head，观察当前隐藏状态对应的词分布，是定位模型知识流动的重要工具。
class LogitLens:
    """
    Applies the LM head at the output of each hidden layer, then analyzes the
    resultant token probability distribution.

    Only works when hooking outputs of *one* individual generation.

    Inspiration: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

    Warning: when running multiple times (e.g. generation), will return
    outputs _only_ for the last processing step.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        layer_module_tmp: str,
        ln_f_module: str,
        lm_head_module: str,
        disabled: bool = False,
    ):
        # 参数说明:
        #   model/tok: 任意AutoModelForCausalLM + AutoTokenizer组合
        #   layer_module_tmp: 字符串模板，用于生成层名并注册Trace钩子，例如"transformer.h.{}"
        #   ln_f_module/lm_head_module: 最终LayerNorm与输出层的模块名，适配GPT样模型
        #   disabled: 允许提前禁用，以简化调用端逻辑（无需条件分支）
        self.disabled = disabled
        self.model, self.tok = model, tok
        self.n_layers = self.model.config.n_layer

        self.lm_head, self.ln_f = (
            nethook.get_module(model, lm_head_module),
            nethook.get_module(model, ln_f_module),
        )

        self.output: Optional[Dict] = None
        self.td: Optional[nethook.TraceDict] = None
        self.trace_layers = [
            layer_module_tmp.format(layer) for layer in range(self.n_layers)
        ]

    def __enter__(self):
        if not self.disabled:
            # TraceDict会在forward期间捕获每层输出；retain_output=True表示保存模块输出张量
            self.td = nethook.TraceDict(
                self.model,
                self.trace_layers,
                retain_input=False,
                retain_output=True,
            )
            self.td.__enter__()

    def __exit__(self, *args):
        if self.disabled:
            return
        self.td.__exit__(*args)

        self.output = {layer: [] for layer in range(self.n_layers)}

        with torch.no_grad():
            for layer, (_, t) in enumerate(self.td.items()):
                cur_out = t.output[0]
                assert (
                    cur_out.size(0) == 1
                ), "Make sure you're only running LogitLens on single generations only."

                # 对每层的最后一个token表示执行LayerNorm与LMHead，得到词表概率
                self.output[layer] = torch.softmax(
                    self.lm_head(self.ln_f(cur_out[:, -1, :])), dim=1
                )

        return self.output

    def pprint(self, k=5):
        # 展示每一层Top-k预测结果，返回(k个token, 概率)元组列表
        to_print = defaultdict(list)

        for layer, pred in self.output.items():
            rets = torch.topk(pred[0], k)
            for i in range(k):
                to_print[layer].append(
                    (
                        self.tok.decode(rets[1][i]),
                        round(rets[0][i].item() * 1e2) / 1e2,
                    )
                )

        print(
            "\n".join(
                [
                    f"{layer}: {[(el[0], round(el[1] * 1e2)) for el in to_print[layer]]}"
                    for layer in range(self.n_layers)
                ]
            )
        )
