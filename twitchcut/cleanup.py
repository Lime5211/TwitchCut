"""Автоочистка: через N дней (по умолчанию 2) у старых задач удаляются тяжёлые файлы — скачанное видео,
звук, готовые клипы. Остаётся всё лёгкое и полезное: названия, оценки, расшифровка, превью, статистика
и отметки «выложил/не подходит» (data/feedback.json не трогается никогда).
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from .util import log, read_json, write_json

HEAVY_DIRS = ("segments", "live_audio", "transcript_parts")
HEAVY_GLOBS = ("audio_src.*", "clips/*.mp4", "clips/*.rendering.mp4", "*.part", "*.ytdl")
ACTIVE = ("running", "queued", "awaiting_llm", "live")


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def disk_usage(ws: Path) -> dict:
    total = heavy = 0
    for d in ws.iterdir() if ws.is_dir() else []:
        if d.is_dir():
            total += _size(d)
            heavy += sum(_size(d / x) for x in HEAVY_DIRS if (d / x).exists())
            heavy += sum(_size(f) for g in HEAVY_GLOBS for f in d.glob(g))
    return {"total": total, "heavy": heavy}


def cleanup(ws: Path, keep_days: float, busy: set | None = None) -> dict:
    """Удаляет тяжёлые файлы задач старше keep_days. busy — id задач, которые сейчас в работе."""
    now = time.time()
    freed, jobs = 0, []
    if keep_days <= 0 or not ws.is_dir():
        return {"freed": 0, "jobs": []}
    for d in ws.iterdir():
        jp = d / "job.json"
        if not d.is_dir() or not jp.exists() or d.name in (busy or set()):
            continue
        st = read_json(jp) or {}
        if st.get("status") in ACTIVE:
            continue
        last = max(float(st.get("updated") or 0), float(st.get("created") or 0), jp.stat().st_mtime)
        if now - last < keep_days * 86400:
            continue
        has = any((d / x).exists() for x in HEAVY_DIRS) or any(True for g in HEAVY_GLOBS for _ in d.glob(g))
        if not has:
            continue
        got = 0
        for x in HEAVY_DIRS:
            p = d / x
            if p.exists():
                got += _size(p)
                shutil.rmtree(p, ignore_errors=True)
        for g in HEAVY_GLOBS:
            for f in d.glob(g):
                try:
                    got += f.stat().st_size
                    f.unlink()
                except OSError:
                    pass
        st.update(files_deleted=True, files_deleted_at=now)
        write_json(jp, st)
        freed += got
        jobs.append(d.name)
        log.info("Очистка: %s — удалено %.0f МБ (старше %g дн.)", d.name, got / 1e6, keep_days)
    return {"freed": freed, "jobs": jobs}


def start_background(get_ws, get_days, get_busy, interval: float = 6 * 3600) -> None:
    def loop():
        time.sleep(60)
        while True:
            try:
                cleanup(get_ws(), float(get_days()), get_busy())
            except Exception as e:
                log.warning("Автоочистка: %s", e)
            time.sleep(interval)
    threading.Thread(target=loop, name="cleanup", daemon=True).start()
