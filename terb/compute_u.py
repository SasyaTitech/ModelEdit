# 该模块实现ROME论文中用于构造秩一更新左向量u的若干子例程，主要包含协方差逆矩阵的缓存与词表示的提取逻辑。
import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util.globals import *

from .layer_stats import layer_stats
from .rome_hparams import ROMEHyperParams

# 缓存变量: inv_mom2_cache[(模型名称, 层标识)] -> torch.Tensor，形状为[d_hidden, d_hidden]的二阶矩逆矩阵
inv_mom2_cache = {}


def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    # 该函数对应于ROME论文中使用统计数据对隐藏表示进行预条件化的步骤
    # 输入:
    #   model/tok: transformers库提供的自回归语言模型与其分词器
    #   layer_name: 需要读取统计信息的模块名称(如model.layers.{l}.mlp.down_proj)
    #   mom2_dataset/mom2_n_samples/mom2_dtype: 指定二阶矩统计的来源、样本规模与精度
    #   hparams: ROME超参数对象，提供stats_dir等路径
    # 输出: Σ^{-1}，torch.Tensor，设备为CUDA，维度等于该层隐藏单元数

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams
        )
        # torch.inverse计算方阵的数学逆矩阵，这里对二阶矩(Σ)取逆得Σ^{-1}
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to("cuda")
        ).float()  # Cast back to float32

    return inv_mom2_cache[key]


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    # 该函数计算秩一更新中的左向量u，维度为[d_hidden]，对应论文中的输入方向
    print("Computing left vector (u)...")

    # Compute projection token
    # word_repr_args封装repr_tools所需的参数，module_template决定监听哪个子模块，track="in"表示提取该模块输入
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")
        
        # repr_tools.get_reprs_at_word_tokens会在上下文模板中填入主语，并返回指定子token的隐藏表示
        # 返回张量形状为[len(context_templates), d_hidden]
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)  # 对多个上下文模板求平均以稳定估计

    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        # repr_tools.get_reprs_at_idxs基于绝对索引提取token表示，此处索引为每个上下文的最后一个token
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        # 根据论文附录，将隐藏表示乘以Σ^{-1}可减小协方差方向上的偏差
        u = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            hparams=hparams,
        ) @ u.unsqueeze(1)  # u.unsqueeze(1)形状[d_hidden, 1]，矩阵乘法后再去掉冗余维度
        u = u.squeeze()

    # L2归一化，确保u的范数为1，便于与右向量v组成秩一矩阵
    return u / u.norm()
