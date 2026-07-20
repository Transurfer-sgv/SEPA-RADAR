# -*- coding: utf-8 -*-
"""KRX 전종목 일봉 수집·캐시 (pykrx).

설계:
- 캐시: cache/ohlcv/YYYYMMDD.parquet — 하루 1파일(전종목 스냅샷).
  매일 실행 시 '빠진 영업일'만 추가 수집 → 일일 증분 1~2콜.
- 최초 백필: 약 2년(500영업일) × 1콜 ≈ 15~25분. GitHub Actions 캐시로 유지.
- 영업일 달력: 삼성전자(005930) 개별 일봉의 인덱스를 달력 소스로 사용.

주의: 이 모듈은 오프라인 환경에서 라이브 검증되지 않았다(첫 로컬 실행 시 확인).
pykrx API 시그니처가 버전에 따라 다를 수 있어 두 가지 호출 경로를 시도한다.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PKG_ROOT / "cache" / "ohlcv"   # 기본: sepa-radar/cache/ohlcv


def set_cache_dir(p) -> None:
    """캐시 디렉터리 재지정(통합 모드에서 config.json paths.cache_dir 반영용)."""
    global CACHE_DIR
    CACHE_DIR = Path(p)
KOR2ENG = {"시가": "open", "고가": "high", "저가": "low", "종가": "close",
           "거래량": "volume", "거래대금": "value"}


def _snapshot(date_str: str) -> pd.DataFrame:
    """해당 일자의 전종목(KOSPI+KOSDAQ) OHLCV. 빈 df면 휴장일."""
    from pykrx import stock
    frames = []
    for mkt in ("KOSPI", "KOSDAQ"):
        try:
            df = stock.get_market_ohlcv_by_ticker(date_str, market=mkt)
        except TypeError:
            df = stock.get_market_ohlcv(date_str, market=mkt)
        if df is not None and len(df):
            df = df.rename(columns=KOR2ENG)
            df["market"] = mkt
            frames.append(df[[c for c in ["open", "high", "low", "close", "volume", "value", "market"] if c in df.columns]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out.index.name = "ticker"
    return out.reset_index()


def trading_days(start: str, end: str) -> list[str]:
    from pykrx import stock
    ref = stock.get_market_ohlcv(start, end, "005930")
    return [d.strftime("%Y%m%d") for d in ref.index]


def ensure_cache(start: str, end: str, sleep_sec: float = 0.4, log=print) -> list[str]:
    """start~end 영업일 스냅샷을 캐시에 채우고, 캐시된 일자 목록 반환."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    days = trading_days(start, end)
    have = {p.stem for p in CACHE_DIR.glob("*.parquet")}
    missing = [d for d in days if d not in have]
    log(f"[data] 영업일 {len(days)}일, 캐시 {len(days)-len(missing)}일, 수집 필요 {len(missing)}일")
    for i, d in enumerate(missing, 1):
        snap = _snapshot(d)
        if len(snap):
            snap.to_parquet(CACHE_DIR / f"{d}.parquet", index=False)
        if i % 20 == 0:
            log(f"[data] {i}/{len(missing)} … {d}")
        time.sleep(sleep_sec)
    return [d for d in days if (CACHE_DIR / f"{d}.parquet").exists()]


def load_panel(days: list[str]) -> pd.DataFrame:
    """캐시 → long 패널(date, ticker, open..volume, market)."""
    frames = []
    for d in days:
        f = CACHE_DIR / f"{d}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            df["date"] = pd.Timestamp(d)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ticker_names(tickers: list[str]) -> dict[str, str]:
    from pykrx import stock
    out = {}
    for t in tickers:
        try:
            out[t] = stock.get_market_ticker_name(t)
        except Exception:
            out[t] = t
    return out


def common_stock_universe(panel: pd.DataFrame, names: dict[str, str],
                          min_close: float, min_value20: float) -> list[str]:
    """보통주만: 티커 끝자리 0, 스팩 제외 + 가격/유동성 필터."""
    last_day = panel["date"].max()
    recent = panel[panel["date"] >= last_day - pd.Timedelta(days=45)]
    val20 = recent.groupby("ticker")["value"].mean() if "value" in recent.columns else None
    last = panel[panel["date"] == last_day].set_index("ticker")
    out = []
    for t in last.index:
        if not str(t).endswith("0"):
            continue
        nm = names.get(t, "")
        if "스팩" in nm:
            continue
        if float(last.loc[t, "close"]) < min_close:
            continue
        if val20 is not None and t in val20.index and float(val20.loc[t]) < min_value20:
            continue
        out.append(t)
    return out
