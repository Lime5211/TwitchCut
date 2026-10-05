"""Статистика выложенных роликов TikTok по ссылке: просмотры, лайки, комментарии, репосты.

Работает без входа в аккаунт: сначала через yt-dlp (он понимает и короткие ссылки vm.tiktok.com),
если не вышло — читает данные прямо со страницы видео. Обновляется в фоне: свежие ролики (до 7 дней)
раз в 3 часа, старые — раз в сутки. История просмотров сохраняется, чтобы видеть динамику.
"""
from __future__ import annotations

import json
import re
import threading
import time

import requests

from .util import TwitchCutError, log

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0.0.0 Safari/537.36")
_lock = threading.Lock()


def is_tiktok(url: str) -> bool:
    return bool(re.search(r"(^|\.)tiktok\.com/", url or ""))


def _via_ytdlp(url: str) -> dict:
    import yt_dlp
    class _Quiet:  # yt-dlp печатает ERROR в консоль даже с quiet — глушим, ошибка и так обрабатывается
        def debug(self, msg): pass
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "socket_timeout": 20,
                           "logger": _Quiet()}) as y:
        info = y.extract_info(url, download=False)
    if not info or info.get("view_count") is None:
        raise TwitchCutError("yt-dlp не вернул просмотры")
    return {
        "views": info.get("view_count"), "likes": info.get("like_count"), "comments": info.get("comment_count"),
        "shares": info.get("repost_count"), "saves": info.get("save_count"),
        "video_id": str(info.get("id") or ""), "posted_ts": info.get("timestamp"),
        "tt_duration": info.get("duration"), "tt_desc": (info.get("description") or info.get("title") or "")[:300],
        "url_full": info.get("webpage_url") or url,
    }


def _via_page(url: str) -> dict:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "ru,en;q=0.8"})
    r = s.get(url, timeout=20, allow_redirects=True)
    r.raise_for_status()
    html = r.text
    m = re.search(r'<script[^>]+id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', html, re.S)
    stats = None
    item = {}
    if m:
        try:
            data = json.loads(m.group(1))
            item = data["__DEFAULT_SCOPE__"]["webapp.video-detail"]["itemInfo"]["itemStruct"]
            stats = item.get("statsV2") or item.get("stats")
        except (ValueError, KeyError, TypeError):
            stats = None
    if not stats:
        def num(key):
            mm = re.search(rf'"{key}"\s*:\s*"?(\d+)', html)
            return int(mm.group(1)) if mm else None
        stats = {"playCount": num("playCount"), "diggCount": num("diggCount"), "commentCount": num("commentCount"),
                 "shareCount": num("shareCount"), "collectCount": num("collectCount")}
    if stats.get("playCount") is None:
        raise TwitchCutError("TikTok не отдал статистику (видео удалено, приватное или страница изменилась)")

    def i(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    return {"views": i(stats.get("playCount")), "likes": i(stats.get("diggCount")),
            "comments": i(stats.get("commentCount")), "shares": i(stats.get("shareCount")),
            "saves": i(stats.get("collectCount")), "video_id": str(item.get("id") or ""),
            "posted_ts": i(item.get("createTime")), "tt_desc": (item.get("desc") or "")[:300], "url_full": r.url}


def fetch(url: str) -> dict:
    """Статистика одного ролика. Бросает TwitchCutError, если не удалось."""
    if not is_tiktok(url):
        raise TwitchCutError("Это не ссылка на TikTok")
    errors = []
    # страница видео — быстрее и сейчас надёжнее; yt-dlp — запасной путь
    for fn in (_via_page, _via_ytdlp):
        try:
            res = fn(url)
            return {k: v for k, v in res.items() if v is not None and v != ""}
        except Exception as e:  # пробуем следующий способ
            errors.append(f"{fn.__name__}: {e}")
    raise TwitchCutError("Не удалось получить статистику TikTok: " + "; ".join(errors)[:400])


def due(rec: dict, now: float | None = None) -> bool:
    """Пора ли обновлять: свежие ролики — раз в 3 часа, старше недели — раз в сутки, старше 2 месяцев — раз в неделю."""
    if rec.get("status") != "posted" or not is_tiktok(rec.get("url") or ""):
        return False
    now = now or time.time()
    last = rec.get("stats_at") or 0
    if rec.get("stats_error") and now - (rec.get("stats_err_at") or 0) < 6 * 3600:
        return False
    age = now - (rec.get("posted_ts") or rec.get("posted_at") or now)
    period = 3 * 3600 if age < 7 * 86400 else 86400 if age < 60 * 86400 else 7 * 86400
    return now - last >= period


def apply(rec: dict, st: dict) -> dict:
    """Записывает статистику в отметку и дописывает историю просмотров."""
    rec = {**rec, **st, "stats_at": time.time(), "stats_auto": True}
    rec.pop("stats_error", None)
    full = st.get("url_full") or ""
    if "/@_/" in (rec.get("url") or "") and is_tiktok(full) and "/@_/" not in full:
        rec["url"] = full.split("?")[0]  # временная ссылка без ника -> настоящая
    hist = list(rec.get("history") or [])
    if st.get("views") is not None and (not hist or hist[-1][1] != st["views"]):
        hist.append([int(time.time()), int(st["views"])])
    rec["history"] = hist[-60:]
    return rec


def refresh(force_ids: set | None = None) -> dict:
    """Обновляет статистику всех выложенных роликов, которым пора. Возвращает {обновлено, ошибок}."""
    from . import feedback
    if not _lock.acquire(blocking=False):
        return {"updated": 0, "errors": 0, "busy": True}
    try:
        ok = err = 0
        todo = []
        for rec in feedback.load():
            if force_ids is not None:
                if rec.get("id") not in force_ids or rec.get("status") != "posted" or not is_tiktok(rec.get("url") or ""):
                    continue
            elif not due(rec):
                continue
            todo.append(rec)
        # ролики своих аккаунтов с доступом video.list — пачкой напрямую из API TikTok
        api_done = set()
        try:
            from . import tiktok_upload as tu
            by_acc: dict[str, list] = {}
            for rec in todo:
                if rec.get("tt_open_id") and rec.get("video_id") and tu.has_list(rec["tt_open_id"]):
                    by_acc.setdefault(rec["tt_open_id"], []).append(rec)
            for oid, recs in by_acc.items():
                try:
                    vids = tu.query_videos(oid, [r["video_id"] for r in recs])
                except Exception as e:
                    log.info("TikTok API: %s", e)
                    continue
                for rec in recs:
                    v = vids.get(str(rec["video_id"]))
                    if v:
                        feedback.replace(apply(rec, tu.video_stats(v)))
                        api_done.add(rec["id"])
                        ok += 1
        except Exception as e:
            log.info("TikTok API статистика: %s", e)
        for rec in todo:
            if rec["id"] in api_done:
                continue
            try:
                st = fetch(rec["url"])
                feedback.replace(apply(rec, st))
                ok += 1
            except Exception as e:
                log.info("TikTok: %s — %s", rec.get("url"), e)
                feedback.replace({**rec, "stats_error": str(e)[:200], "stats_err_at": time.time()})
                err += 1
            time.sleep(2)  # не спамим TikTok
        if ok or err:
            log.info("Статистика TikTok: обновлено %d, ошибок %d", ok, err)
        return {"updated": ok, "errors": err}
    finally:
        _lock.release()


def start_background(interval: float = 1800) -> None:
    def loop():
        time.sleep(30)
        while True:
            try:
                refresh()
            except Exception as e:
                log.warning("Обновление статистики TikTok: %s", e)
            time.sleep(interval)
    threading.Thread(target=loop, name="tiktok-stats", daemon=True).start()
