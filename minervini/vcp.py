# -*- coding: utf-8 -*-
"""VCP(Volatility Contraction Pattern) 탐지와 매수 타이밍 판정.

미너비니 기준의 코드화(근사):
- 수축 2~6회, 각 수축 깊이는 직전보다 축소(허용오차 1회), 첫 수축 ≤ 35%
- 최종 수축 ≤ 12%(타이트), 베이스 기간 ≥ 최소 일수
- 거래량 드라이업: 직전 5일 평균 / 50일 평균 ≤ 기준
- 피벗 = 최종 수축의 고점. 매수 신호 = 종가가 피벗 상향 돌파 + 거래량 50일 평균의 배수 이상
- 피벗 +5% 초과 = 추격 금지(EXTENDED)

주의: 종가 확정(EOD) 기준이므로 신호는 장 마감 후 확정된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "lookback": 252,          # 베이스 탐색 구간(거래일)
    "min_contractions": 2,
    "max_contractions": 6,
    "max_first_depth": 0.35,  # 첫 수축(베이스 깊이) 상한
    "max_last_depth": 0.12,   # 최종 수축 상한(타이트)
    "shrink_tol": 1.15,       # 직전 수축 대비 허용 배수(1회 위반 허용)
    "min_base_days": 20,      # 베이스 최소 기간(거래일, 약 4주)
    "vdu_max": 0.85,          # 드라이업: 5일 평균거래량 / 50일 평균 상한
    "vol_breakout_ratio": 1.5,
    "near_pivot_pct": 0.03,   # 피벗 -3% 이내 = 임박
    "buy_range_pct": 0.05,    # 피벗 +5% 이내 = 매수 가능 범위
}


@dataclass
class Contraction:
    hi_i: int
    hi: float
    lo_i: int
    lo: float

    @property
    def depth(self) -> float:
        return (self.hi - self.lo) / self.hi if self.hi > 0 else np.nan


@dataclass
class VCPResult:
    valid: bool = False
    status: str = "NONE"           # NONE/FORMING/NEAR/BREAKOUT/BREAKOUT_WEAK/BUY_RANGE/EXTENDED
    pivot: float | None = None
    pivot_date: str | None = None
    depths: list = field(default_factory=list)
    base_days: int = 0
    vdu: float | None = None
    vol_ratio: float | None = None
    dist_to_pivot_pct: float | None = None
    notes: list = field(default_factory=list)
    contractions: list = field(default_factory=list)  # [(hi_date, hi, lo_date, lo)]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "status": self.status,
            "pivot": None if self.pivot is None else round(float(self.pivot), 2),
            "pivot_date": self.pivot_date,
            "depths": [round(float(d) * 100, 1) for d in self.depths],
            "footprint": "-".join(f"{d*100:.0f}%" for d in self.depths) + (f" {len(self.depths)}T" if self.depths else ""),
            "base_days": int(self.base_days),
            "vdu": None if self.vdu is None else round(float(self.vdu), 2),
            "vol_ratio": None if self.vol_ratio is None else round(float(self.vol_ratio), 2),
            "dist_to_pivot_pct": None if self.dist_to_pivot_pct is None else round(float(self.dist_to_pivot_pct), 2),
            "notes": self.notes,
            "contractions": self.contractions,
        }


def zigzag(high: np.ndarray, low: np.ndarray, th: float):
    """(index, 'H'|'L', price) 피벗 목록과 마지막 추세 상태를 반환."""
    n = len(high)
    piv: list[tuple[int, str, float]] = []
    if n < 3:
        return piv, 0, (0, float(high[0]) if n else 0.0)
    hi_i, hi_p = 0, float(high[0])
    lo_i, lo_p = 0, float(low[0])
    trend = 0
    ext_i, ext_p = 0, float(high[0])
    i = 1
    while trend == 0 and i < n:
        if high[i] > hi_p:
            hi_i, hi_p = i, float(high[i])
        if low[i] < lo_p:
            lo_i, lo_p = i, float(low[i])
        if hi_p >= lo_p * (1 + th):
            if hi_i > lo_i:
                piv.append((lo_i, "L", lo_p))
                trend, ext_i, ext_p = 1, hi_i, hi_p
            else:
                piv.append((hi_i, "H", hi_p))
                trend, ext_i, ext_p = -1, lo_i, lo_p
        i += 1
    while i < n:
        if trend == 1:
            if high[i] > ext_p:
                ext_i, ext_p = i, float(high[i])
            if low[i] <= ext_p * (1 - th):
                piv.append((ext_i, "H", ext_p))
                trend, ext_i, ext_p = -1, i, float(low[i])
        else:
            if low[i] < ext_p:
                ext_i, ext_p = i, float(low[i])
            if high[i] >= ext_p * (1 + th):
                piv.append((ext_i, "L", ext_p))
                trend, ext_i, ext_p = 1, i, float(high[i])
        i += 1
    return piv, trend, (ext_i, ext_p)


def _adaptive_threshold(df: pd.DataFrame) -> float:
    """매크로 수축 판별용 지그재그 임계값(5~10%).

    너무 낮으면 노이즈가 수축으로 잡혀 큰 수축(20%+)이 분절된다.
    최종 타이트 수축(<임계값)은 detect_vcp의 '진행 중 수축' 로직이 별도로 잡는다.
    """
    rng = ((df["high"] - df["low"]) / df["close"]).tail(60)
    med = float(np.nanmedian(rng)) if len(rng) else 0.03
    return float(min(0.10, max(0.05, 1.5 * med)))


def detect_vcp(df: pd.DataFrame, params: dict | None = None) -> VCPResult:
    """지표 포함 일봉 df에서 마지막 봉 기준 VCP/피벗 상태를 판정."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    res = VCPResult()
    if len(df) < 60:
        res.notes.append("데이터 부족")
        return res

    w = df.tail(int(p["lookback"])).reset_index()
    date_col = w.columns[0]
    high = w["high"].to_numpy(dtype=float)
    low = w["low"].to_numpy(dtype=float)
    close = w["close"].to_numpy(dtype=float)
    vol = w["volume"].to_numpy(dtype=float)
    vol50 = w["vol50"].to_numpy(dtype=float)
    n = len(w)

    th = _adaptive_threshold(df)
    # 구조(수축·피벗)는 '전일까지'로 판정한다.
    # 오늘의 돌파 신고가가 베이스 저항 구조를 지워버리는 것을 막고,
    # "패턴이 먼저, 돌파는 그 패턴에 대한 판정"이라는 시간 순서를 지키기 위함.
    piv, trend, (ext_i, ext_p) = zigzag(high[:-1], low[:-1], th)

    # H→L 쌍으로 수축 후보 구성(확정 피벗)
    pairs: list[Contraction] = []
    k = 0
    while k < len(piv) - 1:
        if piv[k][1] == "H" and piv[k + 1][1] == "L":
            pairs.append(Contraction(piv[k][0], piv[k][2], piv[k + 1][0], piv[k + 1][2]))
            k += 2
        else:
            k += 1
    # 진행 중(미확정) 최종 수축: 마지막 확정 H, 또는 상승 진행 중이면 미확정 고점 기준.
    # 단, 미확정 고점이 기존 저항(확정 수축 고점 최대치)을 2% 초과하면
    # 이미 돌파 이후 국면이므로 확정 구조까지만 사용한다(BUY_RANGE/EXTENDED 판정용).
    resistance = max((c.hi for c in pairs), default=None)
    last_h_i, last_h_p = None, None
    if piv and piv[-1][1] == "H":
        last_h_i, last_h_p = piv[-1][0], piv[-1][2]
    elif trend == 1 and (resistance is None or ext_p <= resistance * 1.02):
        last_h_i, last_h_p = ext_i, ext_p
    if last_h_i is not None:
        seg_lo_i = last_h_i + int(np.argmin(low[last_h_i:]))  # 오늘 저가 포함
        d = 1.0 - float(low[seg_lo_i]) / last_h_p if last_h_p else 0.0
        # 깊이 1% 미만·2봉 미만의 '수축'은 단순 추세 진행이므로 베이스로 보지 않음
        if d >= 0.01 and (len(low) - 1 - last_h_i) >= 2:
            pairs.append(Contraction(last_h_i, last_h_p, seg_lo_i, float(low[seg_lo_i])))

    if len(pairs) < 1:
        res.notes.append("수축 구조 없음")
        return res

    # 베이스 정합: 고점이 사실상 내려오거나 유지되는 접미 구간만 채택(저항선 아래 수축)
    kept = [pairs[-1]]
    for cnt in reversed(pairs[:-1]):
        if cnt.hi >= kept[0].hi * 0.98:
            kept.insert(0, cnt)
        else:
            break
    kept = kept[-int(p["max_contractions"]):]

    depths = [c.depth for c in kept]
    final = kept[-1]
    pivot = float(final.hi)
    base_days = int(n - 1 - kept[0].hi_i)
    last_close = float(close[-1])
    prev_close = float(close[-2]) if n >= 2 else last_close
    v50 = vol50[-1] if np.isfinite(vol50[-1]) and vol50[-1] > 0 else np.nan
    vol_ratio = float(vol[-1] / v50) if np.isfinite(v50) else None
    pre = vol[max(0, n - 6):n - 1]
    vdu = float(np.mean(pre) / v50) if np.isfinite(v50) and len(pre) else None

    res.pivot = pivot
    res.pivot_date = str(pd.Timestamp(w.loc[final.hi_i, date_col]).date())
    res.depths = [float(d) for d in depths]
    res.base_days = base_days
    res.vdu = vdu
    res.vol_ratio = vol_ratio
    res.dist_to_pivot_pct = (last_close / pivot - 1) * 100
    res.contractions = [
        [str(pd.Timestamp(w.loc[c.hi_i, date_col]).date()), round(c.hi, 2),
         str(pd.Timestamp(w.loc[c.lo_i, date_col]).date()), round(c.lo, 2)]
        for c in kept
    ]

    # 유효성 검사
    checks = []
    n_c = len(kept)
    checks.append((p["min_contractions"] <= n_c <= p["max_contractions"], f"수축 횟수 {n_c}회"))
    checks.append((depths[0] <= p["max_first_depth"], f"베이스 깊이 {depths[0]*100:.0f}%"))
    viol = sum(1 for a, b in zip(depths, depths[1:]) if b > a * p["shrink_tol"])
    checks.append((viol <= 1, f"수축 축소 위반 {viol}회"))
    checks.append((depths[-1] <= p["max_last_depth"], f"최종 수축 {depths[-1]*100:.1f}%"))
    checks.append((base_days >= p["min_base_days"], f"베이스 {base_days}일"))
    res.valid = all(ok for ok, _ in checks)
    res.notes = [("✓ " if ok else "✗ ") + msg for ok, msg in checks]
    if vdu is not None:
        dry = vdu <= p["vdu_max"]
        res.notes.append(("✓ " if dry else "△ ") + f"드라이업 {vdu:.2f} (기준 ≤{p['vdu_max']})")

    if not res.valid:
        res.status = "NONE"
        return res

    # 상태 판정(종가 기준)
    if last_close > pivot * (1 + p["buy_range_pct"]):
        res.status = "EXTENDED"
    elif prev_close <= pivot < last_close:
        if vol_ratio is not None and vol_ratio >= p["vol_breakout_ratio"]:
            res.status = "BREAKOUT"
        else:
            res.status = "BREAKOUT_WEAK"
    elif last_close > pivot:
        res.status = "BUY_RANGE"
    elif last_close >= pivot * (1 - p["near_pivot_pct"]):
        res.status = "NEAR"
    else:
        res.status = "FORMING"
    return res
