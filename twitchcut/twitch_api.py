"""Мини-клиент Twitch GQL.

Twitch требует «Client-Integrity» токен для запросов с Client-Id веб-сайта. Запросы с Client-Id
мобильного приложения эту проверку не требуют, поэтому он идёт первым, а веб-ID — запасным.
"""
from __future__ import annotations

import random
import time

import requests

from .util import TwitchCutError, log

GQL_URL = "https://gql.twitch.tv/gql"
CLIENT_IDS = [
    "kd1unb4b3q4t58fwlpcbzcbnm76a8fp",  # мобильное приложение — без integrity-проверки
    "kimne78kx3ncx6brgo4mv6wki5h1ko",   # веб-плеер
]
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
]

_good_client: dict = {}


class IntegrityError(TwitchCutError):
    pass


class GQL:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": random.choice(_USER_AGENTS), "Accept-Language": "ru-RU"})

    def _post(self, client_id: str, body, auth: str | None = None) -> dict:
        last = None
        hdr = {"Client-Id": client_id}
        if auth:
            hdr["Authorization"] = f"OAuth {auth}"
        for attempt in range(5):
            try:
                r = self.s.post(GQL_URL, json=body, headers=hdr, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    data = data[0] if isinstance(data, list) else data
                    errs = data.get("errors") or []
                    if errs:
                        msg = "; ".join(str(e.get("message")) for e in errs)
                        if "integrity" in msg.lower():
                            raise IntegrityError(msg)
                        if "service" in msg.lower() or "timeout" in msg.lower():
                            last = msg
                            time.sleep(1.0 * (attempt + 1))
                            continue
                        raise TwitchCutError(f"Twitch: {msg}")
                    return data
                last = f"HTTP {r.status_code}"
            except (requests.RequestException, ValueError) as e:
                last = str(e)
            time.sleep(1.0 * (attempt + 1))
        raise TwitchCutError(f"Twitch не отвечает: {last}")

    def request(self, body, prefer_web: bool = False, auth: str | None = None) -> dict:
        """Пробует Client-Id по очереди; запоминает тот, что сработал."""
        if auth:  # токен аккаунта привязан к Client-Id веб-сайта
            return self._post(CLIENT_IDS[1], body, auth=auth)
        if prefer_web:  # токены воспроизведения стабильнее выдаются веб-клиенту
            order = list(reversed(CLIENT_IDS))
        else:
            order = ([_good_client["id"]] if "id" in _good_client else []) + \
                    [c for c in CLIENT_IDS if c != _good_client.get("id")]
        last_err: Exception | None = None
        for cid in order:
            try:
                data = self._post(cid, body)
                if not prefer_web:
                    _good_client["id"] = cid
                return data
            except IntegrityError as e:
                log.debug("Client-Id %s… отклонён проверкой целостности", cid[:6])
                last_err = e
        raise TwitchCutError(f"Twitch отклонил запрос (integrity): {last_err}")

    def query(self, q: str, variables: dict | None = None, prefer_web: bool = False, auth: str | None = None) -> dict:
        return self.request({"query": q, "variables": variables or {}}, prefer_web=prefer_web, auth=auth)

    def persisted(self, op: str, sha: str, variables: dict) -> dict:
        return self.request([{"operationName": op, "variables": variables,
                              "extensions": {"persistedQuery": {"version": 1, "sha256Hash": sha}}}])


# ------------------------------------------------------------- high level
def channel_info(login: str) -> dict:
    q = """query($login: String!) { user(login: $login) {
        id login displayName description profileImageURL(width: 150)
        stream { id title viewersCount game { name } }
    } }"""
    d = GQL().query(q, {"login": login.lower()})
    u = (d.get("data") or {}).get("user")
    if not u:
        raise TwitchCutError(f"Канал «{login}» не найден на Twitch")
    stream = u.get("stream")
    return {
        "login": u["login"], "name": u.get("displayName") or u["login"], "twitch_id": u.get("id"),
        "avatar": u.get("profileImageURL") or "", "bio": u.get("description") or "",
        "live": bool(stream), "live_title": (stream or {}).get("title") or "",
        "live_game": ((stream or {}).get("game") or {}).get("name") or "",
        "live_viewers": (stream or {}).get("viewersCount") or 0,
    }


def channel_vods(login: str, limit: int = 20) -> list[dict]:
    q = """query($login: String!, $first: Int!) { user(login: $login) {
        videos(first: $first, type: ARCHIVE, sort: TIME) { edges { node {
            id title lengthSeconds createdAt viewCount previewThumbnailURL(width: 320, height: 180)
            game { name }
        } } }
    } }"""
    d = GQL().query(q, {"login": login.lower(), "first": limit})
    u = (d.get("data") or {}).get("user")
    if not u:
        raise TwitchCutError(f"Канал «{login}» не найден")
    out = []
    for e in ((u.get("videos") or {}).get("edges") or []):
        n = e.get("node") or {}
        out.append({
            "id": n.get("id"), "title": n.get("title") or "", "duration": n.get("lengthSeconds") or 0,
            "created": n.get("createdAt") or "", "views": n.get("viewCount") or 0,
            "thumb": n.get("previewThumbnailURL") or "", "game": (n.get("game") or {}).get("name") or "",
            "url": f"https://www.twitch.tv/videos/{n.get('id')}",
        })
    return out


def live_status(login: str) -> dict:
    """В эфире ли канал и какая запись (VOD) пишется прямо сейчас."""
    q = """query($login: String!) { user(login: $login) {
        login displayName
        stream { id createdAt title game { name } }
        videos(first: 1, type: ARCHIVE, sort: TIME) { edges { node { id status lengthSeconds createdAt title } } }
    } }"""
    d = GQL().query(q, {"login": login.lower()})
    u = (d.get("data") or {}).get("user")
    if not u:
        raise TwitchCutError(f"Канал «{login}» не найден")
    stream = u.get("stream")
    edges = ((u.get("videos") or {}).get("edges")) or []
    vod = (edges[0].get("node") if edges else None) or None
    live_vod = None
    if stream and vod:
        # запись текущего эфира: статус RECORDING или создана после начала эфира
        if (vod.get("status") or "").upper() == "RECORDING" or (vod.get("createdAt") or "") >= (stream.get("createdAt") or "z"):
            live_vod = vod
    return {"live": bool(stream), "stream_title": (stream or {}).get("title") or "",
            "game": ((stream or {}).get("game") or {}).get("name") or "",
            "started": (stream or {}).get("createdAt") or "", "vod": live_vod}


def vod_status(vod_id: str) -> dict:
    q = """query($id: ID!) { video(id: $id) { id status lengthSeconds title owner { login } } }"""
    d = GQL().query(q, {"id": str(vod_id)})
    v = (d.get("data") or {}).get("video")
    if not v:
        raise TwitchCutError("Запись не найдена (возможно, удалена)")
    return {"id": v["id"], "status": (v.get("status") or "").upper(), "length": float(v.get("lengthSeconds") or 0),
            "title": v.get("title") or "", "login": ((v.get("owner") or {}).get("login")) or ""}


# ------------------------------------------------------------ аккаунт Twitch
def _data_dir():
    from .util import ROOT_DIR
    return ROOT_DIR / "data"


def cookies_path():
    return _data_dir() / "twitch_cookies.txt"


def get_auth_token() -> str:
    """Токен аккаунта Twitch: из data/twitch_auth.json или из cookies (auth-token)."""
    from .util import read_json
    d = read_json(_data_dir() / "twitch_auth.json") or {}
    if d.get("auth_token"):
        return d["auth_token"]
    cp = cookies_path()
    if cp.exists():
        for line in cp.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 7 and parts[5] == "auth-token":
                return parts[6].strip()
    return ""


def save_auth(text: str) -> dict:
    """Принимает cookies в формате Netscape (как из расширения «Get cookies.txt») или просто auth-token."""
    import re as _re
    from .util import write_json
    text = (text or "").strip()
    token = ""
    is_cookies = "\t" in text and ".twitch.tv" in text
    if is_cookies:
        for line in text.splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 7 and parts[5] == "auth-token":
                token = parts[6].strip()
    elif _re.fullmatch(r"[a-z0-9]{20,40}", text):
        token = text
    if not token:
        raise TwitchCutError("Не нашёл auth-token. Вставьте cookies twitch.tv целиком (формат Netscape) "
                             "или только значение auth-token.")
    if is_cookies:
        cookies_path().write_text(text + "\n", encoding="utf-8")
    info = auth_info(token)
    write_json(_data_dir() / "twitch_auth.json", {"auth_token": token, "login": info.get("login", "")})
    return info


def clear_auth() -> None:
    for p in (_data_dir() / "twitch_auth.json", cookies_path()):
        p.unlink(missing_ok=True)


def auth_info(token: str | None = None) -> dict:
    token = token if token is not None else get_auth_token()
    if not token:
        return {"ok": False, "login": "", "error": "Аккаунт не подключён"}
    try:
        d = GQL().query("query { currentUser { login displayName } }", {}, auth=token)
        u = (d.get("data") or {}).get("currentUser")
        if not u:
            return {"ok": False, "login": "", "error": "Токен не принят Twitch (устарел?)"}
        return {"ok": True, "login": u.get("login"), "name": u.get("displayName")}
    except Exception as e:
        return {"ok": False, "login": "", "error": str(e)}
