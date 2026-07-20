# -*- coding: utf-8 -*-
"""이동평균·52주 고저·트렌드 템플릿(8조건) 계산.

기준 출처: Mark Minervini, "Trade Like a Stock Market Wizard" (2013)
- Trend Template 8조건을 그대로 코드화. 6번(52주 저가 대비 +30%),
  7번(52주 고가 대비 -25% 이내), 8번(RS 백분위 70 이상)은 책의 수치 사용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """일봉 DataFrame(date index, open/high/low/close/volume)에 지표 추가."""
    out = df.copy()
    c = out["close"].astype(float)
    out["ma50"] = c.rolling(50).mean()
    out["ma150"] = c.rolling(150).mean()
    out["ma200"] = c.rolling(200).mean()
    out["vol50"] = out["volume"].rolling(50).mean()
    # 52주(252거래일) 고저 — 장중 고가/저가 기준
    out["hi52"] = out["high"].rolling(252, min_periods=60).max()
    out["lo52"] = out["low"].rolling(252, min_periods=60).min()
    # 200일선 기울기: 1개월(21거래일) 전 대비
    out["ma200_slope1m"] = out["ma200"] - out["ma200"].shift(21)
    # 5개월(105거래일) 전 대비 — 참고용(책: "최소 1개월, 가급적 4~5개월 상승")
    out["ma200_slope5m"] = out["ma200"] - out["ma200"].shift(105)
    return out


TEMPLATE_LABELS = {
    "T1": "주가 > 150일선 & 200일선",
    "T2": "150일선 > 200일선",
    "T3": "200일선 1개월 이상 상승",
    "T4": "50일선 > 150일선 & 200일선",
    "T5": "주가 > 50일선",
    "T6": "52주 저가 대비 +30% 이상",
    "T7": "52주 고가 대비 25% 이내",
    "T8": "RS 백분위 70 이상",
}


def _f(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def evaluate_template(row: pd.Series, rs_pct: float | None) -> dict:
    """마지막 봉 기준 트렌드 템플릿 8조건 평가.

    반환: {"conds": {T1..T8: bool}, "passed": int, "total": 8,
           "values": 표시용 수치, "rs": rs_pct}
    NaN(데이터 부족)은 False 처리.
    """
    c = _f(row.get("close"))
    ma50, ma150, ma200 = _f(row.get("ma50")), _f(row.get("ma150")), _f(row.get("ma200"))
    slope1m = _f(row.get("ma200_slope1m"))
    hi52, lo52 = _f(row.get("hi52")), _f(row.get("lo52"))
    rs = None if rs_pct is None or not np.isfinite(_f(rs_pct)) else float(rs_pct)

    def ok(cond: bool, *vals) -> bool:
        return bool(cond) and all(np.isfinite(v) for v in vals)

    conds = {
        "T1": ok(c > ma150 and c > ma200, c, ma150, ma200),
        "T2": ok(ma150 > ma200, ma150, ma200),
        "T3": ok(slope1m > 0, slope1m),
        "T4": ok(ma50 > ma150 and ma50 > ma200, ma50, ma150, ma200),
        "T5": ok(c > ma50, c, ma50),
        "T6": ok(c >= lo52 * 1.30, c, lo52),
        "T7": ok(c >= hi52 * 0.75, c, hi52),
        "T8": rs is not None and rs >= 70,
    }
    values = {
        "close": c,
        "ma50": ma50,
        "ma150": ma150,
        "ma200": ma200,
        "ma200_slope1m": slope1m,
        "pct_above_lo52": (c / lo52 - 1) * 100 if np.isfinite(lo52) and lo52 > 0 else None,
        "pct_from_hi52": (c / hi52 - 1) * 100 if np.isfinite(hi52) and hi52 > 0 else None,
        "rs": rs,
    }
    return {
        "conds": conds,
        "passed": int(sum(conds.values())),
        "total": 8,
        "values": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 2)) for k, v in values.items()},
        "rs": rs,
    }
