"""Сохранённые стримеры: описание, мемы, свои эмоуты и настройки монтажа для каждого канала.

Хранятся в data/streamers.json. Описание стримера передаётся Claude при каждом анализе —
это главный способ научить систему понимать, что смешно именно у этого стримера.
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from .util import ROOT_DIR, TwitchCutError, log, read_json, write_json

PATH = ROOT_DIR / "data" / "streamers.json"
_lock = threading.Lock()

FIELDS = {
    "login": "", "name": "", "avatar": "", "bio": "",
    "description": "",      # кто он, во что играет, характер, мемы, коронные фразы
    "funny_emotes": "",     # эмоуты канала, которые значат «смешно» (через пробел или запятую)
    "hype_emotes": "",      # эмоуты «круто/вау»
    "layout": "auto",       # auto | split | crop | blur
    "cam_rect": "",         # "x,y,w,h" в долях кадра, если автоопределение вебки ошибается
    "max_clips": 0,         # 0 = как в настройках
    "language": "",         # пусто = как в настройках
    "watch": False,         # следить за эфиром и нарезать прямо во время стрима
    "cam_auto": None,       # где вебка — найдено автоматически по прошлым стримам (заполняется само)
    "learned_taste": "",    # что заходит у канала — выучено по отметкам автора (обновляется само)
    "learned_edit": "",     # пожелания к монтажу — выучено по замечаниям автора
    "learned_n": 0,
    "tiktok_account": "",   # open_id аккаунта TikTok, куда отправлять клипы этого стримера
    "tag_ru": "",           # второй хештег в описании TikTok — как стримера зовут зрители (стинт, тоха, дрейк)
    "learned_at": 0,
    "payout_views": 0,      # сколько просмотров нужно ролику, чтобы за него заплатили (0 — не отслеживать)
}

# пороги выплат по умолчанию (можно поменять на странице «Выплаты»)
DEFAULT_PAYOUT = {"t2x2": 200_000, "stintik": 200_000, "drakeoffc": 50_000}


def normalize_login(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"twitch\.tv/([A-Za-z0-9_]{2,25})", s)
    if m:
        s = m.group(1)
    s = s.lstrip("@").lower()
    if not re.fullmatch(r"[a-z0-9_]{2,25}", s):
        raise TwitchCutError("Логин канала: латиница, цифры и _, например «buster» или ссылка twitch.tv/buster")
    return s


def load_all() -> list[dict]:
    data = read_json(PATH) or []
    out = []
    for d in data:
        if isinstance(d, dict) and d.get("login"):
            rec = {**FIELDS, **d}
            if "payout_views" not in d:
                rec["payout_views"] = DEFAULT_PAYOUT.get(d["login"], 0)
            out.append(rec)
    return out


def get(login: str | None) -> dict | None:
    if not login:
        return None
    login = login.lower()
    return next((s for s in load_all() if s["login"] == login), None)


def upsert(data: dict, fetch_profile: bool = True) -> dict:
    login = normalize_login(data.get("login") or "")
    with _lock:
        items = load_all()
        cur = next((s for s in items if s["login"] == login), None)
        rec = {**FIELDS, **(cur or {}), **{k: v for k, v in data.items() if k in FIELDS}}
        rec["login"] = login
        rec["watch"] = bool(rec.get("watch")) and str(rec.get("watch")).lower() not in ("false", "0", "")
        try:
            rec["max_clips"] = int(rec.get("max_clips") or 0)
        except (TypeError, ValueError):
            rec["max_clips"] = 0
        try:
            rec["payout_views"] = max(0, int(str(rec.get("payout_views") or 0).replace(" ", "")))
        except (TypeError, ValueError):
            rec["payout_views"] = 0
        if rec.get("cam_rect"):
            parse_cam_rect(rec["cam_rect"])  # валидация
        if fetch_profile and (not cur or not rec.get("avatar")):
            try:
                from .twitch_api import channel_info
                info = channel_info(login)
                rec["name"] = rec.get("name") or info["name"]
                rec["avatar"] = info["avatar"]
                rec["bio"] = info["bio"]
            except Exception as e:
                log.warning("Профиль %s не получен: %s", login, e)
        rec["name"] = rec.get("name") or login
        rec["updated"] = time.time()
        rec.setdefault("created", time.time())
        items = [s for s in items if s["login"] != login] + [rec]
        items.sort(key=lambda s: s["name"].lower())
        write_json(PATH, items)
    return rec


def set_fields(login: str, **fields) -> None:
    """Записать служебные поля стримера (найденная вебка, выученный вкус канала)."""
    with _lock:
        items = read_json(PATH) or []
        for d in items:
            if isinstance(d, dict) and d.get("login") == login.lower():
                d.update(fields)
                write_json(PATH, items)
                return


def set_cam_auto(login: str, cam: dict | None) -> None:
    """Запомнить, где у стримера вебка (по клипам последнего стрима) — подсказка для следующих нарезок."""
    if not login:
        return
    with _lock:
        items = read_json(PATH) or []
        for d in items:
            if isinstance(d, dict) and d.get("login") == login.lower():
                d["cam_auto"] = cam
                write_json(PATH, items)
                return


def delete(login: str) -> None:
    with _lock:
        items = [s for s in load_all() if s["login"] != login.lower()]
        write_json(PATH, items)


def parse_cam_rect(s) -> list[float] | None:
    if not s:
        return None
    if isinstance(s, (list, tuple)):
        vals = [float(v) for v in s]
    else:
        vals = [float(v) for v in re.split(r"[,\s;]+", str(s).strip()) if v]
    if len(vals) != 4 or not all(0 <= v <= 1 for v in vals):
        raise TwitchCutError("Вебка: 4 числа от 0 до 1 через запятую — x, y, ширина, высота. Например 0.75, 0.7, 0.25, 0.3")
    return vals


def emote_list(s: str) -> list[str]:
    return [e for e in re.split(r"[,\s]+", s or "") if e]


def apply_to_config(st: dict, cfg: dict, extra_context: str = "") -> dict:
    """Переносит настройки стримера в конфиг задачи."""
    ctx = []
    head = f"Стример: {st.get('name') or st['login']} (twitch.tv/{st['login']})."
    ctx.append(head)
    if st.get("description"):
        ctx.append(st["description"].strip())
    if st.get("funny_emotes") or st.get("hype_emotes"):
        ctx.append("Эмоуты канала: смешно — " + (st.get("funny_emotes") or "—")
                   + "; круто — " + (st.get("hype_emotes") or "—") + ".")
    if st.get("learned_taste"):
        ctx.append("ЧТО ЗАХОДИТ У ЭТОГО КАНАЛА (выводы по отметкам автора и просмотрам):\n" + st["learned_taste"].strip())
    if extra_context:
        ctx.append(extra_context.strip())
    cfg["llm"]["streamer_context"] = "\n".join(ctx)
    cfg["llm"]["streamer_login"] = st["login"]
    if st.get("layout") and st["layout"] != "auto":
        cfg["render"]["layout"] = st["layout"]
    if st.get("cam_rect"):
        # раскладка остаётся «авто»: вебка известна, а экран/вебка на весь экран выбираются по кадрам
        cfg["render"]["cam_rect"] = parse_cam_rect(st["cam_rect"])
    if st.get("cam_auto"):
        cfg["render"]["cam_prior"] = st["cam_auto"]
    if st.get("max_clips"):
        cfg["clips"]["max_clips"] = int(st["max_clips"])
    if st.get("language"):
        cfg["language"] = st["language"]
    from . import lexicon
    lexicon.add_custom("funny", emote_list(st.get("funny_emotes", "")))
    lexicon.add_custom("hype", emote_list(st.get("hype_emotes", "")))
    return cfg


GENERIC_TAGS = ("твич", "стрим")


def tiktok_caption(login: str, hashtags: list | None, st: dict | None = None) -> str:
    """Описание для TikTok: «twitch: <ник> #<ник> #<прозвище> + 3 хештега по теме ролика (или #твич #стрим)»."""
    import re
    login = (login or "").strip().lower()
    st = st or get(login) or {}
    ru = re.sub(r"[#\s]+", "", st.get("tag_ru") or "")
    head = [t for t in (login, ru) if t]
    skip = {t.lower() for t in head} | {"twitch", "твич", "стрим", "stream", "стример", "нарезки", "нарезка",
                                         re.sub(r"\s+", "", (st.get("name") or "")).lower()}
    extra: list[str] = []
    for h in hashtags or []:
        h = re.sub(r"[#\s.,!?]+", "", str(h))
        if h and h.lower() not in skip and h.lower() not in {e.lower() for e in extra}:
            extra.append(h)
    extra = extra[:3]
    for g in GENERIC_TAGS:
        if len(extra) >= 3:
            break
        if g not in extra:
            extra.append(g)
    return (f"twitch: {login} " if login else "") + " ".join("#" + t for t in head + extra)
