import numpy as np
import math

def encode_state(payload):
    """將攻擊向量字串轉換為 Byte Histogram 特徵表示。"""
    # 初始化256維頻率向量
    freq = np.zeros(256, dtype=float)
    L = len(payload)
    for ch in payload:
        ascii_val = ord(ch) if ord(ch) < 256 else 255  # 非ASCII字元統一算入255
        freq[ascii_val] += 1
    if L > 0:
        freq = freq / L  # 計算每個字元出現頻率
    # 將長度 L 與頻率串接，構成 state 向量
    state_vec = np.concatenate(([L], freq))
    return state_vec
