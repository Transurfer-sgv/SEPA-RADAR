# -*- coding: utf-8 -*-
"""데모용 샘플 데이터 생성 + 프런트엔드 임베드.

실제 분석 코드(run_analysis)를 그대로 통과시켜 산출하므로
단계 분류·VCP 판정 로직의 스모크 테스트를 겸한다.
실행: python tools/make_sample_data.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_daily import run_analysis, OUT  # noqa: E402

RNG = np.random.default_rng(20260719)
N = 560
DATES = pd.bdate_range(end="2026-07-17", periods=N)

NAMES = {
    "005930": "삼성전자", "196170": "알테오젠", "950160": "코오롱티슈진",
    "000660": "SK하이닉스", "042700": "한미반도체", "403870": "HPSP",
}


def _mk_df(close: np.ndarray, vol: np.ndarray) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    spread = np.abs(RNG.normal(0.008, 0.004, N)) * close
    high = close + spread
    low = np.maximum(close - spread, close * 0.9)
    opn = np.clip(close * (1 + RNG.normal(0, 0.004, N)), low, high)
    return pd.DataFrame({"open": opn, "high": high, "low": low,
                         "close": close, "volume": np.maximum(vol, 1000).astype(int)},
                        index=DATES)


def _seg(start: float, end: float, days: int, vol_pct: float) -> np.ndarray:
    path = np.geomspace(start, end, days)
    noise = RNG.normal(0, vol_pct, days).cumsum()
    noise -= np.linspace(0, noise[-1], days)  # 끝점 고정
    return path * np.exp(noise)


def _concat(*parts: np.ndarray) -> np.ndarray:
    out = np.concatenate(parts)
    assert len(out) == N, len(out)
    return out


def series_stage2_vcp(base: float, breakout: bool) -> pd.DataFrame:
    """완만 상승 → 가속 상승 → VCP(22→13→7→3%) → 임박 또는 돌파.

    2단 가속 구조로 52주 저가가 가속 초입에 형성되어 T6(+30%)를 충족한다.
    """
    up1 = _seg(base, base * 1.25, 290, 0.010)          # 완만한 1차 상승
    up2 = _seg(up1[-1], base * 2.60, 100, 0.013)       # 가속(주도주 국면)
    pivot = base * 2.60 * 1.02
    run = _seg(up2[-1], pivot, 24, 0.004)              # 피벗(베이스 고점)
    c1d = _seg(pivot, pivot * 0.78, 20, 0.011)         # 수축1 -22%
    c1u = _seg(c1d[-1], pivot * 0.995, 22, 0.009)
    c2d = _seg(c1u[-1], pivot * 0.865, 13, 0.008)      # 수축2 -13%
    c2u = _seg(c2d[-1], pivot * 0.990, 16, 0.007)
    c3d = _seg(c2u[-1], pivot * 0.925, 10, 0.006)      # 수축3 -7%
    c3u = _seg(c3d[-1], pivot * 0.985, 12, 0.005)
    c4d = _seg(c3u[-1], pivot * 0.958, 8, 0.004)       # 수축4 -3% 타이트
    tail_n = N - (290 + 100 + 24 + 20 + 22 + 13 + 16 + 10 + 12 + 8)
    if breakout:
        tail = _seg(c4d[-1], pivot * 0.975, tail_n - 1, 0.003)
        tail = np.append(tail, pivot * 1.030)          # 마지막 날 돌파 마감
    else:
        tail = _seg(c4d[-1], pivot * 0.980, tail_n, 0.003)  # 피벗권 임박
    close = _concat(up1, up2, run, c1d, c1u, c2d, c2u, c3d, c3u, c4d, tail)

    v = np.full(N, 1.0)
    v[:414] = RNG.uniform(0.8, 1.6, 414)
    v[300:390] *= 1.6                                  # 가속 구간 거래량 확대
    v[414:] = np.linspace(1.1, 0.55, N - 414) * RNG.uniform(0.85, 1.15, N - 414)
    v[-6:-1] = 0.45                                    # 돌파 직전 드라이업
    v[-1] = 2.3 if breakout else 0.5
    return _mk_df(close, v * 8e5)


def series_stage2_strong(base: float) -> pd.DataFrame:
    """강한 2단계 지속(신고가권, 피벗 없음/EXTENDED 성격)."""
    a = _seg(base, base * 1.4, 200, 0.014)
    b = _seg(a[-1], a[-1] * 2.4, 300, 0.016)
    c = _seg(b[-1], b[-1] * 1.10, 60, 0.012)
    return _mk_df(_concat(a, b, c), RNG.uniform(0.7, 1.8, N) * 6e5)


def series_stage4(base: float) -> pd.DataFrame:
    """4단계 하락(-55%)."""
    a = _seg(base, base * 1.15, 140, 0.014)
    b = _seg(a[-1], a[-1] * 0.42, 340, 0.020)
    c = _seg(b[-1], b[-1] * 0.96, 80, 0.016)
    return _mk_df(_concat(a, b, c), RNG.uniform(0.6, 1.7, N) * 1.2e6)


def series_stage3(base: float) -> pd.DataFrame:
    """장기 상승 후 천장권 요동(200일선 위 넓은 스윙, 평탄화)."""
    a = _seg(base, base * 2.3, 470, 0.013)
    peak = a[-1]
    b = _seg(peak, peak * 0.91, 25, 0.016)
    c1 = _seg(b[-1], peak * 0.98, 15, 0.014)   # 고점 재테스트 실패
    c2 = _seg(c1[-1], peak * 0.89, 15, 0.014)
    c3 = _seg(c2[-1], peak * 0.96, 15, 0.014)
    c4 = _seg(c3[-1], peak * 0.87, 20, 0.014)
    close = _concat(a, b, c1, c2, c3, c4)
    v = RNG.uniform(0.8, 2.0, N) * 9e5
    v[470:] *= 1.4                                 # 분산 국면 거래량 확대
    return _mk_df(close, v)


def series_filler(i: int) -> pd.DataFrame:
    drift = RNG.uniform(-0.55, 0.35)
    start = RNG.uniform(3000, 90000)
    close = _seg(start, start * (1 + drift), N, 0.018)
    return _mk_df(close, RNG.uniform(0.5, 1.5, N) * 4e5)


def main():
    panel_map = {
        "005930": series_stage2_vcp(33000, breakout=False),   # 보유 · VCP 임박
        "196170": series_stage2_strong(190000),               # 보유 · 강한 2단계
        "950160": series_stage4(21000),                       # 보유 · 4단계
        "000660": series_stage3(98000),                       # 보유 · 3단계
        "042700": series_stage2_vcp(38000, breakout=True),    # 신호 · 오늘 돌파
        "403870": series_stage2_vcp(23000, breakout=False),   # 신호 · 임박
    }
    names = dict(NAMES)
    for i in range(120):
        t = f"9{i:05d}"
        panel_map[t] = series_filler(i)
        names[t] = f"샘플{i:03d}"

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data_date = DATES[-1].strftime("%Y-%m-%d")
    # 이전 상태 파일 제거(변화 알림 오염 방지)
    for f in (OUT / "holdings.json",):
        if f.exists():
            f.unlink()
    result = run_analysis(panel_map, names, cfg, data_date, sample=True)

    print("\n=== 스모크 테스트 결과 ===")
    holdings = json.loads((OUT / "holdings.json").read_text(encoding="utf-8"))
    for h in holdings:
        print(f"{h['name']}({h['ticker']}): {h['stage']['label']} | "
              f"템플릿 {h['template']['passed']}/8 | VCP {h['vcp']['status']} {h['vcp']['footprint']}")
    print(f"돌파 {len(result['breakout'])} / 임박 {len(result['near'])}")
    for s in result["breakout"] + result["near"]:
        print(f"  [{s['status']}] {s['name']} pivot {s['pivot']:.0f} dist {s['dist_to_pivot_pct']:.2f}% vol× {s['vol_ratio']}")

    # 프런트엔드 임베드(오프라인·아티팩트 미리보기 폴백)
    embed = {
        "meta": json.loads((OUT / "meta.json").read_text(encoding="utf-8")),
        "holdings": holdings,
        "signals": json.loads((OUT / "signals.json").read_text(encoding="utf-8")),
        "ohlcv": {p.stem: json.loads(p.read_text(encoding="utf-8"))
                  for p in (OUT / "ohlcv").glob("*.json")},
    }
    html_path = ROOT / "docs" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    blob = json.dumps(embed, ensure_ascii=False, separators=(",", ":"))
    new = re.sub(r"/\*__SAMPLE_START__\*/.*?/\*__SAMPLE_END__\*/",
                 "/*__SAMPLE_START__*/window.__SAMPLE__=" + blob + ";/*__SAMPLE_END__*/",
                 html, flags=re.S)
    html_path.write_text(new, encoding="utf-8")
    print(f"\n임베드 완료: index.html ({len(blob)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
