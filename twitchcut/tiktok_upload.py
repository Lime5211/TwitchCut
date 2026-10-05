"""Отправка клипа в черновики TikTok одной кнопкой (официальный TikTok Content Posting API, режим «Upload»).

Клип приходит в приложение TikTok во «Входящие» (уведомление «Видео готово к публикации»): там вы пишете
подпись, при желании добавляете звук и публикуете. Сам TwitchCut ничего не публикует.

Нужно один раз:
1. На developers.tiktok.com создать приложение (платформа Desktop), добавить продукты Login Kit и
   Content Posting API, scope video.upload, redirect URI http://127.0.0.1:8765/tiktok/callback/
2. В режиме Sandbox добавить свой аккаунт TikTok в Target users (тогда проверка приложения не нужна).
3. Вставить Client key и Client secret на странице «Статус» и войти в каждый аккаунт TikTok
   (у каждого стримера может быть свой аккаунт — клип уходит в аккаунт своего стримера).

Ограничения TikTok: не больше 5 неопубликованных черновиков за сутки, 6 запросов в минуту.
"""
from __future__ import annotations

import hashlib
import secrets
import string
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

from .util import ROOT_DIR, TwitchCutError, log, read_json, write_json

APP_PATH = ROOT_DIR / "data" / "tiktok_app.json"        # client_key / client_secret (только на этом ПК)
TOKENS_PATH = ROOT_DIR / "data" / "tiktok_tokens.json"  # входы в аккаунты TikTok: {open_id: токены} (только на этом ПК)
OLD_TOKEN_PATH = ROOT_DIR / "data" / "tiktok_token.json"
AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
API = "https://open.tiktokapis.com/v2"
SCOPES = "user.info.basic,video.upload"
REDIRECT = "http://127.0.0.1:{port}/tiktok/callback/"

_pending: dict[str, dict] = {}   # state -> {verifier, redirect, at, streamer}

ERRORS = {
    "spam_risk_too_many_pending_share": "В этом аккаунте TikTok уже 5 неопубликованных черновиков за сутки — опубликуйте или удалите их во «Входящих» и попробуйте снова.",
    "spam_risk_too_many_posts": "TikTok ограничил число загрузок за сутки для этого аккаунта — попробуйте завтра.",
    "access_token_invalid": "Вход в этот аккаунт TikTok устарел — на странице «Статус» войдите в него ещё раз.",
    "scope_not_authorized": "Приложению не дан доступ video.upload — войдите в TikTok заново и разрешите загрузку видео.",
    "rate_limit_exceeded": "Слишком много запросов к TikTok — подождите минуту.",
    "file_format_check_failed": "TikTok не принял файл: формат не подходит.",
}


def app_info() -> dict:
    return read_json(APP_PATH) or {}


def save_app(client_key: str, client_secret: str) -> None:
    client_key, client_secret = (client_key or "").strip(), (client_secret or "").strip()
    if not client_key or not client_secret:
        raise TwitchCutError("Укажите Client key и Client secret приложения TikTok")
    write_json(APP_PATH, {"client_key": client_key, "client_secret": client_secret})


def _tokens() -> dict:
    toks = read_json(TOKENS_PATH)
    if toks is None:
        toks = {}
        old = read_json(OLD_TOKEN_PATH)  # первая версия хранила один аккаунт
        if old and old.get("open_id"):
            toks[old["open_id"]] = old
            write_json(TOKENS_PATH, toks)
    return toks


def accounts() -> list[dict]:
    """Подключённые аккаунты TikTok и к каким стримерам они привязаны."""
    from . import streamers
    by = {}
    for st in streamers.load_all():
        if st.get("tiktok_account"):
            by.setdefault(st["tiktok_account"], []).append(st["login"])
    out = []
    for oid, t in _tokens().items():
        out.append({"open_id": oid, "name": t.get("display_name") or oid[:10], "avatar": t.get("avatar_url"),
                    "ok": bool(t.get("refresh_token")) and t.get("refresh_expires_at", 0) > time.time(),
                    "streamers": by.get(oid, [])})
    return out


def status() -> dict:
    app = app_info()
    acc = accounts()
    return {"configured": bool(app.get("client_key")), "connected": any(a["ok"] for a in acc), "accounts": acc}


def disconnect(open_id: str | None = None) -> None:
    from . import streamers
    toks = _tokens()
    for oid in ([open_id] if open_id else list(toks)):
        toks.pop(oid, None)
        for st in streamers.load_all():
            if st.get("tiktok_account") == oid:
                streamers.set_fields(st["login"], tiktok_account="")
    write_json(TOKENS_PATH, toks)


def login_url(port: int, streamer: str | None = None) -> str:
    app = app_info()
    if not app.get("client_key"):
        raise TwitchCutError("Сначала укажите Client key и Client secret приложения TikTok")
    alphabet = string.ascii_letters + string.digits + "-._~"
    verifier = "".join(secrets.choice(alphabet) for _ in range(64))
    challenge = hashlib.sha256(verifier.encode()).hexdigest()  # TikTok для desktop: hex(SHA256)
    state = secrets.token_urlsafe(16)
    redirect = REDIRECT.format(port=port)
    for k in [k for k, v in _pending.items() if time.time() - v["at"] > 900]:
        _pending.pop(k, None)
    _pending[state] = {"verifier": verifier, "redirect": redirect, "at": time.time(), "streamer": streamer or ""}
    q = {"client_key": app["client_key"], "scope": SCOPES, "redirect_uri": redirect, "state": state,
         "response_type": "code", "code_challenge": challenge, "code_challenge_method": "S256"}
    return AUTH_URL + "?" + urlencode(q)


def _save_token(data: dict, open_id: str | None = None) -> dict:
    if not data.get("access_token"):
        raise TwitchCutError("TikTok не выдал токен: " + str(data.get("error_description") or data.get("error") or data)[:300])
    now = time.time()
    oid = data.get("open_id") or open_id
    toks = _tokens()
    old = toks.get(oid) or {}
    tok = {**old, "access_token": data["access_token"], "refresh_token": data.get("refresh_token") or old.get("refresh_token"),
           "expires_at": now + float(data.get("expires_in") or 86400) - 120,
           "refresh_expires_at": now + float(data.get("refresh_expires_in") or 365 * 86400) - 3600,
           "open_id": oid, "scope": data.get("scope") or old.get("scope")}
    toks[oid] = tok
    write_json(TOKENS_PATH, toks)
    return tok


def finish_login(code: str, state: str) -> dict:
    p = _pending.pop(state or "", None)
    if not p:
        raise TwitchCutError("Вход в TikTok устарел или открыт не из TwitchCut — начните заново со страницы «Статус»")
    app = app_info()
    r = requests.post(TOKEN_URL, data={
        "client_key": app["client_key"], "client_secret": app["client_secret"], "code": code,
        "grant_type": "authorization_code", "redirect_uri": p["redirect"], "code_verifier": p["verifier"]},
        headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
    tok = _save_token(r.json())
    if "video.upload" not in (tok.get("scope") or ""):
        log.warning("TikTok: в выданных правах нет video.upload (%s)", tok.get("scope"))
    try:
        u = requests.get(API + "/user/info/?fields=open_id,display_name,avatar_url",
                         headers={"Authorization": "Bearer " + tok["access_token"]}, timeout=20).json()
        info = (u.get("data") or {}).get("user") or {}
        toks = _tokens()
        toks[tok["open_id"]].update(display_name=info.get("display_name"), avatar_url=info.get("avatar_url"))
        write_json(TOKENS_PATH, toks)
        tok = toks[tok["open_id"]]
    except Exception as e:
        log.info("TikTok: не удалось получить имя аккаунта: %s", e)
    if p.get("streamer"):
        from . import streamers
        streamers.set_fields(p["streamer"], tiktok_account=tok["open_id"])
        tok = {**tok, "streamer": p["streamer"]}
    log.info("TikTok: вход выполнен (%s)", tok.get("display_name") or tok.get("open_id"))
    return tok


def account_for(streamer: str | None) -> str:
    """Какой аккаунт TikTok у стримера. Если аккаунт один — он; иначе нужна привязка в карточке стримера."""
    from . import streamers
    toks = _tokens()
    if not toks:
        raise TwitchCutError("TikTok не подключён — на странице «Статус» нажмите «Войти в TikTok»")
    st = streamers.get(streamer) if streamer else None
    if st and st.get("tiktok_account") in toks:
        return st["tiktok_account"]
    if len(toks) == 1:
        return next(iter(toks))
    raise TwitchCutError(f"Не выбран аккаунт TikTok для стримера {streamer or ''} — выберите его на странице "
                         "«Статус» в блоке TikTok")


def access_token(open_id: str) -> str:
    tok = _tokens().get(open_id) or {}
    if not tok.get("refresh_token"):
        raise TwitchCutError("Этот аккаунт TikTok не подключён — войдите в него на странице «Статус»")
    if tok.get("expires_at", 0) > time.time():
        return tok["access_token"]
    app = app_info()
    r = requests.post(TOKEN_URL, data={"client_key": app.get("client_key"), "client_secret": app.get("client_secret"),
                                       "grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
                      headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
    data = r.json()
    if not data.get("access_token"):
        raise TwitchCutError(f"Вход в аккаунт TikTok {tok.get('display_name') or ''} устарел — войдите в него ещё раз на странице «Статус»")
    return _save_token(data, open_id)["access_token"]


def _check(resp: requests.Response) -> dict:
    try:
        j = resp.json()
    except ValueError:
        raise TwitchCutError(f"TikTok ответил {resp.status_code}: {resp.text[:200]}")
    err = j.get("error") or {}
    code = err.get("code") or ("ok" if resp.status_code == 200 else str(resp.status_code))
    if code != "ok":
        raise TwitchCutError(ERRORS.get(code) or f"TikTok: {code} — {err.get('message') or ''}".strip(" —"))
    return j.get("data") or {}


def _chunks(size: int) -> tuple[int, int]:
    """Размер куска и число кусков по правилам TikTok: до 64 МБ — одним куском, иначе куски по 32 МБ
    (последний кусок забирает остаток, он может быть больше)."""
    if size <= 64 * 1024 * 1024:
        return size, 1
    chunk = 32 * 1024 * 1024
    return chunk, size // chunk


def upload_draft(path: Path, open_id: str, progress=None) -> dict:
    """Загружает видео во «Входящие» аккаунта TikTok open_id. Возвращает {publish_id, status, account}."""
    path = Path(path)
    if not path.is_file():
        raise TwitchCutError("Файл клипа не найден (возможно, удалён автоочисткой — нажмите «Применить», чтобы смонтировать заново)")
    size = path.stat().st_size
    chunk, count = _chunks(size)
    token = access_token(open_id)
    h = {"Authorization": "Bearer " + token, "Content-Type": "application/json; charset=UTF-8"}
    data = _check(requests.post(API + "/post/publish/inbox/video/init/", headers=h, timeout=30, json={
        "source_info": {"source": "FILE_UPLOAD", "video_size": size, "chunk_size": chunk, "total_chunk_count": count}}))
    url, pid = data.get("upload_url"), data.get("publish_id")
    if not url:
        raise TwitchCutError("TikTok не выдал адрес загрузки")
    with open(path, "rb") as f:
        for i in range(count):
            first = i * chunk
            last = size - 1 if i == count - 1 else first + chunk - 1
            f.seek(first)
            body = f.read(last - first + 1)
            r = requests.put(url, data=body, timeout=600, headers={
                "Content-Type": "video/mp4", "Content-Length": str(len(body)),
                "Content-Range": f"bytes {first}-{last}/{size}"})
            if r.status_code not in (200, 201, 206):
                raise TwitchCutError(f"Загрузка в TikTok прервалась ({r.status_code}): {r.text[:200]}")
            if progress:
                progress(i + 1, count)
    st = "PROCESSING_UPLOAD"
    for _ in range(10):  # ждём, пока TikTok примет файл (обычно секунды)
        time.sleep(3)
        try:
            d = _check(requests.post(API + "/post/publish/status/fetch/", headers=h, timeout=20,
                                     json={"publish_id": pid}))
        except TwitchCutError as e:
            log.info("TikTok: статус загрузки пока недоступен: %s", e)
            continue
        st = d.get("status") or st
        if st in ("SEND_TO_USER_INBOX", "PUBLISH_COMPLETE"):
            break
        if st == "FAILED":
            raise TwitchCutError("TikTok не принял видео: " + str(d.get("fail_reason") or "неизвестная причина"))
    log.info("TikTok: %s отправлен в черновики (%s, %s)", path.name, pid, st)
    name = (_tokens().get(open_id) or {}).get("display_name") or open_id[:10]
    return {"publish_id": pid, "status": st, "at": time.time(), "account": name}
