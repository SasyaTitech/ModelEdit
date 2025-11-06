import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 该函数评估任意文本在参考语言模型上的困惑度，常用于编辑前后模型的流畅度对比。
def perplexity(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    max_input_length: int = None,
):
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """

    # 将单条文本编码为batch=1的输入，max_input_length可限制最大token数，避免超出模型上下文窗口
    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")

    # logits形状为[1, seq_len, vocab_size]，对最后一维做log_softmax获得对数概率
    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    # torch.gather按照实际token索引抽取log P(x_t | x_<t)，得到形状[seq_len-1, 1]
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()
