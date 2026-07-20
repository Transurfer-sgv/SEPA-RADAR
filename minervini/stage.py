# -*- coding: utf-8 -*-
"""SEPA 단계 분석(1 기반 다지기 / 2 상승 / 3 천장 / 4 하락) 휴리스틱 분류.

미너비니의 단계 정의는 정성적이므로, 아래는 구조 조건 기반의 근사 분류다.
- 2단계 구조: 주가>150·200일선, 150>200, 200일선 상승(1개월)
- 4단계 구조: 위의 거울상(모두 반대)
- 전환 구간(1·3단계)은 직전 60거래일 구조 이력으로 판별:
  직전이 상승 구조 우세였으면 3단계(천장), 하락 구조 우세였으면 1단계(기반).
한계: 규칙 기반 근사이므로 실제 차트 판독과 다를 수 있음(README 참고).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAGE_LABELS = {
    0: ("판정불가", "데이터 부족(200일선 미형성)"),
    1: ("1단계 · 기반 다지기", "하락 후 바닥 다지기 국면"),
    2: ("2단계 · 상승 추세", "기관 매집이 유효한 매수 가능 국면"),
    3: ("3단계 · 천장 형성", "분산 신호 — 신규 매수 금지, 보유분 관리"),
    4: ("4단계 · 하락 추세", "매도/관망 국면 — 신규 매수 금지"),
}


def _structural_series(df: pd.DataFrame) -> pd.Series:
    c, ma50 = df["close"], df["ma50"]
    ma150, ma200 = df["ma150"], df["ma200"]
    slope = df["ma200_slope1m"]
    up = (c > ma150) & (c > ma200) & (ma150 > ma200) & (slope > 0)
    down = (c < ma150) & (c < ma200) & (ma150 < ma200) & (slope < 0)
    s = pd.Series(0, index=df.index, dtype=float)
    s[up.fillna(False)] = 1.0
    s[down.fillna(False)] = -1.0
    return s


def classify_stage(df: pd.DataFrame) -> dict:
    """지표가 추가된 일봉 df의 '마지막 봉' 기준 단계 분류."""
    row = df.iloc[-1]
    needed = ["ma50", "ma150", "ma200", "ma200_slope1m"]
    if any(pd.isna(row.get(k)) for k in needed):
        label, desc = STAGE_LABELS[0]
        return {"stage": 0, "label": label, "desc": desc, "confidence": 0.0}

    s = _structural_series(df)
    today = s.iloc[-1]

    # 신뢰도: 세부 조건 충족 비율
    c, ma50 = row["close"], row["ma50"]
    ma150, ma200, slope = row["ma150"], row["ma200"], row["ma200_slope1m"]
    up_checks = [c > ma150, c > ma200, ma150 > ma200, slope > 0, c > ma50]
    down_checks = [c < ma150, c < ma200, ma150 < ma200, slope < 0, c < ma50]

    if today > 0:
        stage = 2
        conf = sum(bool(x) for x in up_checks) / len(up_checks)
    elif today < 0:
        stage = 4
        conf = sum(bool(x) for x in down_checks) / len(down_checks)
    else:
        hist = s.iloc[-61:-1]
        m = float(hist.mean()) if len(hist) else 0.0
        if m > 0.15:
            stage = 3
        elif m < -0.15:
            stage = 1
        else:
            stage = 3 if c >= ma200 else 1
        conf = min(0.9, 0.5 + abs(m) / 2)

    label, desc = STAGE_LABELS[stage]
    # 부가 설명
    extra = []
    hi52 = row.get("hi52")
    if stage == 2 and pd.notna(hi52) and hi52 > 0:
        gap = (c / hi52 - 1) * 100
        if gap >= -5:
            extra.append("52주 고가권(-5% 이내)")
    if stage == 2 and c < ma50:
        extra.append("50일선 하회 — 2단계 내 눌림/약화 주의")
    if extra:
        desc = desc + " · " + ", ".join(extra)
    return {"stage": int(stage), "label": label, "desc": desc, "confidence": round(float(conf), 2)}
