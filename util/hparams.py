import json
from dataclasses import dataclass


# HyperParams是所有编辑/微调方法共享的基类，便于将JSON配置直接反序列化为dataclass实例。
@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        # fpath: 指向超参数JSON文件的路径；文件中的键需要与dataclass字段一致
        with open(fpath, "r") as f:
            data = json.load(f)

        # 将JSON映射到构造函数，相当于cls(**data)
        return cls(**data)
