"""Чат VOD: скачивание через Twitch GQL или загрузка JSON из TwitchDownloader.

Формат внутри TwitchCut: список [время_сек, ник, текст].
"""
from __future__ import annotations

from pathlib import Path

from .twitch_api import GQL
from .util import ProgressFn, TwitchCutError, log, noop_progress, read_json

COMMENTS_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"


def _page(gql: GQL, vod_id: str, cursor: str | None, offset: float | None) -> dict:
    variables: dict = {"videoID": vod_id}
    if cursor:
        variables["cursor"] = cursor
    else:
        variables["contentOffsetSeconds"] = int(offset or 0)
    data = gql.persisted("VideoCommentsByOffsetOrCursor", COMMENTS_HASH, variables)
    video = (data.get("data") or {}).get("video")
    if not video:
        raise TwitchCutError("VOD не найден или чат недоступен")
    return video.get("comments") or {}


def _parse(edges: list, out: list) -> float:
    last_t = -1.0
    for e in edges:
        n = e.get("node") or {}
        t = float(n.get("contentOffsetSeconds") or 0)
        user = ((n.get("commenter") or {}).get("displayName")) or ""
        frags = ((n.get("message") or {}).get("fragments")) or []
        text = "".join(f.get("text") or "" for f in frags).strip()
        if text:
            out.append([t, user, text])
        last_t = max(last_t, t)
    return last_t


def fetch_chat_gql(vod_id: str, duration: float, progress: ProgressFn = noop_progress) -> list:
    """Сначала листаем курсором (быстро). Если курсор отклонён — шагаем по смещению времени."""
    gql = GQL()
    messages: list = []
    cursor = None
    last_t = 0.0
    pages = 0
    mode = "cursor"
    while True:
        try:
            comments = _page(gql, vod_id, cursor if mode == "cursor" else None,
                             None if (mode == "cursor" and cursor) else last_t)
        except TwitchCutError:
            if mode == "cursor" and cursor:
                log.info("Курсор чата отклонён — продолжаю по смещению времени")
                mode = "offset"
                continue
            raise
        edges = comments.get("edges") or []
        t = _parse(edges, messages)
        pages += 1
        if duration:
            progress("chat", min(max(t, last_t) / duration, 0.99), f"Скачиваю чат: {len(messages)} сообщений")
        has_next = (comments.get("pageInfo") or {}).get("hasNextPage")
        if not edges or pages > 30000:
            break
        if mode == "cursor":
            if not has_next:
                break
            cursor = edges[-1].get("cursor")
            if not cursor:
                mode = "offset"
            last_t = max(last_t, t)
        else:
            nt = max(last_t + 1, t)
            if (duration and nt >= duration) or (not has_next and nt <= last_t + 1):
                break
            last_t = nt
    seen = set()
    uniq = []
    for m in messages:
        k = (round(m[0], 1), m[1], m[2])
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    uniq.sort(key=lambda m: m[0])
    log.info("Чат: %d сообщений, %d страниц", len(uniq), pages)
    return uniq


def fetch_chat_range(vod_id: str, a: float, b: float) -> list:
    """Чат за отрезок [a, b] (для прямого эфира). Листаем по смещению и курсору до времени b."""
    gql = GQL()
    out: list = []
    comments = _page(gql, vod_id, None, max(0, a))
    pages = 0
    while True:
        edges = comments.get("edges") or []
        last = _parse(edges, out)
        pages += 1
        if not edges or last >= b or not (comments.get("pageInfo") or {}).get("hasNextPage") or pages > 3000:
            break
        cur = edges[-1].get("cursor")
        try:
            comments = _page(gql, vod_id, cur, None) if cur else _page(gql, vod_id, None, last + 1)
        except TwitchCutError:
            comments = _page(gql, vod_id, None, last + 1)
    return sorted([m for m in out if a <= m[0] <= b], key=lambda m: m[0])


def load_chat_file(path: Path) -> list:
    """Поддерживает JSON TwitchDownloaderCLI (`chatdownload`) и собственный формат TwitchCut."""
    data = read_json(path)
    if data is None:
        raise TwitchCutError(f"Не удалось прочитать файл чата {path}")
    if isinstance(data, list):
        return [[float(m[0]), str(m[1]), str(m[2])] for m in data]
    out = []
    for c in data.get("comments", []):
        t = float(c.get("content_offset_seconds") or 0)
        user = (c.get("commenter") or {}).get("display_name") or ""
        msg = c.get("message") or {}
        text = msg.get("body")
        if text is None:
            text = "".join(f.get("text") or "" for f in msg.get("fragments") or [])
        if text:
            out.append([t, user, text.strip()])
    out.sort(key=lambda m: m[0])
    return out
