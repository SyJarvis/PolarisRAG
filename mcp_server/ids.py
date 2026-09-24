# -*- coding: utf-8 -*-
"""ID 生成：Milvus chunk 主键与 source_id。

主键构成：毫秒时间戳 << 21 | 21 位随机数。
生成器保证进程内严格不重复（同毫秒内单调递增），跨进程由
Milvus 主键约束兜底。
"""
import secrets
import time

_last_value = 0


def next_chunk_id() -> int:
    """生成全局唯一的 int64 主键（进程内严格递增去重）。"""
    global _last_value
    while True:
        candidate = (int(time.time() * 1000) << 21) | secrets.randbits(21)
        if candidate > _last_value:
            _last_value = candidate
            return candidate
        # 同毫秒且随机位不大于上次：重取。极小概率重试，毫秒翻转必退出。


def new_source_id() -> str:
    """生成 source 标识："src_" + 6 位随机 hex。"""
    return f"src_{secrets.token_hex(3)}"
