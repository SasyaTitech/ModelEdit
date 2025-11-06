from pathlib import Path

import yaml

# 该模块集中加载项目全局路径配置，供不同编辑算法共享，避免硬编码路径。
with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

# REMOTE_ROOT_URL用于下载预生成统计或缓存文件，可根据部署环境指向HTTP/OSS等存储。
REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
