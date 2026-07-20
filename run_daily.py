# -*- coding: utf-8 -*-
"""SEPA RADAR 일일 파이프라인.

사용:
  python run_daily.py                # 최근 2년 캐시 보장 후 분석·산출(docs/data/)
  python run_daily.py --days 550    # 백필 기간(영업일 환산 아님, 달력일) 조정

산출:
  docs/data/meta.json / holdings.json / signals.json / ohlcv/{ticker}.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from minervini import data as krx
from minervini import telegram
from minervini.indicators import add_indicators, evaluate_template, TEMPLATE_LABELS
from minervini.rs import rs_percentiles
from minervini.stage import classify_stage
from minervini.vcp import detect_vcp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "data"   # 기본값(단독 리포 모드)


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q)


def apply_paths(cfg: dict) -> None:
    """config.json의 paths 설정 반영.

    통합 모드 예: {"docs_data_dir": "../docs/sepa-radar/data", "cache_dir": "cache/ohlcv"}
    상대 경로는 sepa-radar 폴더(ROOT) 기준으로 해석한다.
    """
    global OUT
    paths = cfg.get("paths", {}) or {}
    if paths.get("docs_data_dir"):
        OUT = _resolve(paths["docs_data_dir"])
    if paths.get("cache_dir"):
        krx.set_cache_dir(_resolve(paths["cache_dir"]))


def _jsafe(o):
    if isinstance(o, dict):
        return {k: _jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsafe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsafe(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def analyze_ticker(df: pd.DataFrame, rs: float | None, vcp_params: dict) -> dict:
    """단일 종목 종합 분석(마지막 봉 기준)."""
    dfi = add_indicators(df)
    row = dfi.iloc[-1]
    close = float(row["close"])
    prev = float(dfi["close"].iloc[-2]) if len(dfi) >= 2 else close
    return {
        "close": close,
        "change_pct": (close / prev - 1) * 100 if prev else 0.0,
        "stage": classify_stage(dfi),
        "template": evaluate_template(row, rs),
        "vcp": detect_vcp(dfi, vcp_params).to_dict(),
        "_dfi": dfi,
    }


def export_chart(path: Path, ticker: str, name: str, dfi: pd.DataFrame, analysis: dict, bars: int):
    w = dfi.tail(bars)
    rows = [
        [d.strftime("%Y-%m-%d"), r.open, r.high, r.low, r.close, int(r.volume)]
        for d, r in w.iterrows()
    ]
    def series(col):
        return [None if pd.isna(v) else round(float(v), 2) for v in w[col]]
    payload = {
        "ticker": ticker, "name": name, "rows": rows,
        "ma50": series("ma50"), "ma150": series("ma150"), "ma200": series("ma200"),
        "hi52": None if pd.isna(w["hi52"].iloc[-1]) else float(w["hi52"].iloc[-1]),
        "lo52": None if pd.isna(w["lo52"].iloc[-1]) else float(w["lo52"].iloc[-1]),
        "analysis": {k: v for k, v in analysis.items() if k != "_dfi"},
    }
    _write(path, payload)


def run_analysis(panel_map: dict[str, pd.DataFrame], names: dict[str, str],
                 cfg: dict, data_date: str, sample: bool = False, log=print) -> dict:
    """분석·산출 공통 경로(실데이터·샘플 동일 코드)."""
    sc = cfg["screen"]
    vcp_params = {**sc.get("vcp", {}),
                  "vol_breakout_ratio": sc.get("vol_breakout_ratio", 1.5),
                  "near_pivot_pct": sc.get("near_pivot_pct", 0.03),
                  "buy_range_pct": sc.get("buy_range_pct", 0.05)}
    holdings = [t for t in cfg.get("holdings", []) if t in panel_map]

    log(f"[rs] 유니버스 {len(panel_map)}종목 RS 백분위 계산")
    rs_map = rs_percentiles({t: df["close"] for t, df in panel_map.items()})

    # 이전 상태(보유종목 변화 감지)
    prev_state = {}
    prev_file = OUT / "holdings.json"
    if prev_file.exists():
        try:
            for h in json.loads(prev_file.read_text(encoding="utf-8")):
                prev_state[h["ticker"]] = (h["stage"]["stage"], h["vcp"]["status"])
        except Exception:
            pass

    # 1) 보유종목 전수 분석
    holdings_out, alerts = [], []
    analyses: dict[str, dict] = {}
    for t in holdings:
        a = analyze_ticker(panel_map[t], rs_map.get(t), vcp_params)
        analyses[t] = a
        holdings_out.append({"ticker": t, "name": names.get(t, t),
                             **{k: v for k, v in a.items() if k != "_dfi"}})
        if t in prev_state:
            ps, pv = prev_state[t]
            if ps != a["stage"]["stage"]:
                alerts.append(f"{names.get(t, t)}: {ps}단계 → {a['stage']['stage']}단계")
            if a["vcp"]["status"] in ("BREAKOUT", "NEAR") and pv != a["vcp"]["status"]:
                alerts.append(f"{names.get(t, t)}: VCP {a['vcp']['status']}")

    # 2) 전종목 스크리닝(매수 타이밍)
    breakout, near = [], []
    min_pass, rs_min = sc.get("min_template_pass", 7), sc.get("rs_min", 70)
    cand = 0
    for t, df in panel_map.items():
      try:
        rs = rs_map.get(t)
        if rs is None or rs < rs_min or len(df) < 210:
            continue
        dfi = add_indicators(df)
        tpl = evaluate_template(dfi.iloc[-1], rs)
        if tpl["passed"] < min_pass:
            continue
        stg = classify_stage(dfi)
        if stg["stage"] != 2:
            continue
        cand += 1
        v = detect_vcp(dfi, vcp_params)
        if v.status in ("BREAKOUT", "BREAKOUT_WEAK", "BUY_RANGE", "NEAR"):
            item = {"ticker": t, "name": names.get(t, t),
                    "close": round(float(dfi["close"].iloc[-1]), 2), "pivot": v.pivot,
                    "dist_to_pivot_pct": None if v.dist_to_pivot_pct is None else round(v.dist_to_pivot_pct, 2),
                    "vol_ratio": None if v.vol_ratio is None else round(v.vol_ratio, 2),
                    "rs": rs, "passed": tpl["passed"], "footprint": v.to_dict()["footprint"],
                    "base_days": v.base_days, "status": v.status}
            (near if v.status == "NEAR" else breakout).append(item)
            if t not in analyses:
                analyses[t] = {"close": item["close"],
                               "change_pct": (dfi["close"].iloc[-1] / dfi["close"].iloc[-2] - 1) * 100,
                               "stage": stg, "template": tpl, "vcp": v.to_dict(), "_dfi": dfi}
      except Exception as e:  # noqa: BLE001 — 종목 1개 오류가 전체를 멈추지 않게
        log(f"[screen] {t} 분석 건너뜀: {e}")
        continue
    breakout.sort(key=lambda x: -(x["rs"] or 0))
    near.sort(key=lambda x: -(x["dist_to_pivot_pct"] or -99))
    log(f"[screen] 2단계 후보 {cand} → 돌파 {len(breakout)} / 임박 {len(near)}")

    # 3) 산출
    bars = cfg.get("chart_export_bars", 300)
    chart_dir = OUT / "ohlcv"
    chart_dir.mkdir(parents=True, exist_ok=True)
    for f in chart_dir.glob("*.json"):
        f.unlink()
    export_set = list(dict.fromkeys(holdings + [s["ticker"] for s in breakout + near]))[:60]
    for t in export_set:
        export_chart(chart_dir / f"{t}.json", t, names.get(t, t), analyses[t]["_dfi"], analyses[t], bars)

    _write(OUT / "holdings.json", holdings_out)
    _write(OUT / "signals.json", {"breakout": breakout, "near": near})
    _write(OUT / "meta.json", {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_date": data_date, "universe": len(panel_map),
        "stage2_candidates": cand, "sample": bool(sample),
        "template_labels": TEMPLATE_LABELS,
    })
    return {"breakout": breakout, "near": near, "alerts": alerts, "data_date": data_date}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=760, help="백필 달력일 수(기본 약 2년)")
    ap.add_argument("--no-telegram", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    apply_paths(cfg)
    sc = cfg["screen"]

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=args.days)).strftime("%Y%m%d")
    days = krx.ensure_cache(start, end)
    panel = krx.load_panel(days)
    if panel.empty:
        raise SystemExit("데이터 없음 — pykrx 수집 실패 여부 확인 필요")
    data_date = panel["date"].max().strftime("%Y-%m-%d")

    tickers = sorted(panel["ticker"].unique())
    names = krx.ticker_names(tickers)
    universe = set(krx.common_stock_universe(panel, names,
                                             sc.get("min_close", 1000),
                                             sc.get("min_trading_value_20d", 3e8)))
    universe |= set(cfg.get("holdings", []))

    panel_map = {}
    for t, g in panel[panel["ticker"].isin(universe)].groupby("ticker"):
        g = g.sort_values("date").set_index("date")
        if (g["volume"].tail(20) > 0).any():  # 장기 거래정지 제외
            panel_map[t] = g[["open", "high", "low", "close", "volume"]].astype(float)

    result = run_analysis(panel_map, names, cfg, data_date)

    if not args.no_telegram and cfg.get("telegram", {}).get("enabled", True):
        telegram.send(telegram.format_signals(result["data_date"], result["breakout"],
                                              result["near"], result["alerts"]))


if __name__ == "__main__":
    main()
