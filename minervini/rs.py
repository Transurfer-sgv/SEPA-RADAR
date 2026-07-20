# -*- coding: utf-8 -*-
"""상대강도(RS) 백분위 — IBD식 가중 12개월 수익률의 '근사' 구현.

RS_raw = 0.4×(3개월 수익률) + 0.2×(6개월) + 0.2×(9개월) + 0.2×(12개월)
→ 유니버스 내 백분위(0~100).

주의: IBD 공식 RS Rating과 동일하지 않다(기존 시스템 문서화된 한계와 동일).
상장 1년 미만 종목은 가용 구간만 가중치 재정규화로 계산하며 별도 표기.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [(63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2)]
MIN_HISTORY = 70  # 최소 약 3.5개월


def rs_raw_from_close(close: pd.Series) -> float | None:
    c = close.dropna()
    if len(c) < MIN_HISTORY:
        return None
    last = float(c.iloc[-1])
    score, wsum = 0.0, 0.0
    for win, w in WINDOWS:
        if len(c) > win:
            base = float(c.iloc[-1 - win])
            if base > 0:
                score += w * (last / base - 1.0)
                wsum += w
    if wsum == 0:
        return None
    return score / wsum


def rs_percentiles(close_map: dict[str, pd.Series]) -> dict[str, float]:
    """{티커: 종가시리즈} → {티커: RS 백분위(0~100)}"""
    raw = {t: rs_raw_from_close(s) for t, s in close_map.items()}
    valid = {t: v for t, v in raw.items() if v is not None and np.isfinite(v)}
    if not valid:
        return {}
    ser = pd.Series(valid)
    pct = ser.rank(pct=True) * 100.0
    return {t: round(float(v), 1) for t, v in pct.items()}
