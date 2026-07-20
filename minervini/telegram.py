# -*- coding: utf-8 -*-
"""텔레그램 매수 신호 알림. 환경변수 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 필요.
미설정 시 조용히 건너뛴다.
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request


def send(text: str, log=print) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log("[telegram] 토큰/챗ID 미설정 — 알림 생략")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15) as r:
            ok = r.status == 200
        log(f"[telegram] 전송 {'성공' if ok else '실패'}")
        return ok
    except Exception as e:  # noqa: BLE001
        log(f"[telegram] 오류: {e}")
        return False


def format_signals(data_date: str, breakout: list[dict], near: list[dict], holdings_alerts: list[str]) -> str:
    lines = [f"📡 <b>SEPA RADAR</b> · {data_date}"]
    if breakout:
        lines.append("\n🚀 <b>피벗 돌파(매수 신호)</b>")
        for s in breakout[:10]:
            lines.append(f"· {s['name']}({s['ticker']}) 피벗 {s['pivot']:,.0f} → 종가 {s['close']:,.0f} "
                         f"(+{s['dist_to_pivot_pct']:.1f}%, 거래량 {s['vol_ratio']:.1f}배, RS {s['rs']:.0f})")
    else:
        lines.append("\n🚀 오늘 돌파 신호 없음")
    if near:
        lines.append("\n👀 <b>돌파 임박(피벗 -3% 이내)</b>")
        for s in near[:10]:
            lines.append(f"· {s['name']}({s['ticker']}) 피벗 {s['pivot']:,.0f} (이격 {s['dist_to_pivot_pct']:.1f}%)")
    if holdings_alerts:
        lines.append("\n📌 <b>보유종목 상태 변화</b>")
        lines.extend("· " + a for a in holdings_alerts)
    return "\n".join(lines)
