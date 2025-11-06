from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


# ROMEHyperParams封装论文中的关键超参数，便于在不同模型/任务间复用配置。
@dataclass
class ROMEHyperParams(HyperParams):
    # Method
    # layers: 需要应用秩一编辑的层编号列表，ROMEPaper建议选择靠后的前馈层
    layers: List[int]
    # fact_token: 主语token定位策略，如"subject_last"或"last"，决定u/v提取位置
    fact_token: str
    # v_num_grad_steps: 优化右向量时的梯度迭代次数
    v_num_grad_steps: int
    # v_lr: Adam优化器学习率
    v_lr: float
    # v_loss_layer: 计算语言模型loss的层号，通常与重写层相同或更深
    v_loss_layer: int
    # v_weight_decay: 正则化系数，约束delta的范数
    v_weight_decay: float
    # clamp_norm_factor: 控制delta投影的球半径比例
    clamp_norm_factor: float
    # kl_factor: KL散度损失权重，平衡重写效果与原模型分布
    kl_factor: float
    # mom2_adjustment: 是否启用协方差逆矩阵预条件化
    mom2_adjustment: bool
    # context_template_length_params: 生成上下文模板的长度与数量配置[[长度, 样本数], ...]
    context_template_length_params: List[List[int]]

    # Module templates
    # rewrite_module_tmp: 被编辑线性层的模板字符串，如"model.layers.{}.mlp.down_proj"
    rewrite_module_tmp: str
    # layer_module_tmp: 用于Trace监控loss层的模块模板
    layer_module_tmp: str
    # mlp_module_tmp: 对应MLP块中注入delta的模块模板
    mlp_module_tmp: str
    # attn_module_tmp: 注意力模块模板，部分扩展方法需要
    attn_module_tmp: str
    # ln_f_module: 最终LayerNorm模块名称，用于钩子或分析
    ln_f_module: str
    # lm_head_module: 输出层(词表投影)的模块名称
    lm_head_module: str

    # Statistics
    # mom2_dataset: 收集二阶矩统计所用的数据集名称
    mom2_dataset: str
    # mom2_n_samples: 采样文本数量，用于估计协方差
    mom2_n_samples: int
    # mom2_dtype: 统计文件中存储张量的浮点精度
    mom2_dtype: str
