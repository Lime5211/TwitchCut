"""Источники видео: Twitch VOD (через yt-dlp), любая ссылка yt-dlp, локальный файл.

Ключевая идея экономии: для анализа скачивается только АУДИО (~70 МБ на час стрима),
а видео — только короткие фрагменты вокруг выбранных моментов.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path
from typing import Optional

import requests

from .util import (ProgressFn, TwitchCutError, ffmpeg_bin, ffprobe_duration, fmt_time, log,
                   noop_progress, read_json, run, safe_name, write_json)

TWITCH_VOD_RE = re.compile(r"twitch\.tv/(?:[\w-]+/)?(?:videos?|v)/(\d+)", re.I)


def _cookie_opts() -> dict:
    from .twitch_api import cookies_path
    cp = cookies_path()
    return {"cookiefile": str(cp)} if cp.exists() else {}


def _yt_dlp():
    try:
        import yt_dlp  # noqa
        return yt_dlp
    except ImportError as e:
        raise TwitchCutError("Не установлен yt-dlp: pip install -U yt-dlp") from e


_AUDIO_RE = re.compile(r"^audio_src\.(mp4|m4a|aac|webm|opus|mp3|ts|mkv|flv)$", re.I)


def _audio_files(job_dir: Path) -> list[Path]:
    """Только готовые файлы — без кусков незавершённой загрузки (.part, .part-Frag12, .ytdl)."""
    return sorted(p for p in job_dir.glob("audio_src.*") if _AUDIO_RE.match(p.name) and p.stat().st_size > 0)


def parse_master_playlist(text: str) -> list[dict]:
    """Разбирает master-плейлист HLS Twitch в список вариантов качества."""
    out = []
    names: dict = {}
    lines = [l.strip() for l in text.splitlines()]
    for l in lines:
        if l.startswith("#EXT-X-MEDIA:"):
            g = re.search(r'GROUP-ID="([^"]+)"', l)
            n = re.search(r'NAME="([^"]+)"', l)
            if g and n:
                names[g.group(1)] = n.group(1)
    for i, l in enumerate(lines):
        if not l.startswith("#EXT-X-STREAM-INF:"):
            continue
        url = next((x for x in lines[i + 1:] if x and not x.startswith("#")), None)
        if not url:
            continue
        res = re.search(r"RESOLUTION=(\d+)x(\d+)", l)
        fps = re.search(r"FRAME-RATE=([\d.]+)", l)
        bw = re.search(r"BANDWIDTH=(\d+)", l)
        codecs = (re.search(r'CODECS="([^"]+)"', l) or [None, ""])[1].lower()
        video = (re.search(r'VIDEO="([^"]+)"', l) or [None, ""])[1]
        if not res:  # audio_only
            if "audio" in video.lower() or "mp4a" in codecs:
                out.append({"url": url, "width": 0, "height": 0, "fps": 0.0, "bandwidth": int(bw.group(1)) if bw else 0,
                            "codec": "audio", "name": "audio_only", "headers": {}})
            continue
        codec = "av1" if "av01" in codecs else "hevc" if ("hev1" in codecs or "hvc1" in codecs) else "h264"
        out.append({"url": url, "width": int(res.group(1)), "height": int(res.group(2)),
                    "fps": float(fps.group(1)) if fps else 30.0, "bandwidth": int(bw.group(1)) if bw else 0,
                    "codec": codec, "name": names.get(video, video or f"{res.group(2)}p"), "headers": {}})
    return out


def _parse_media_playlist(text: str, url: str) -> tuple[list, Optional[str]]:
    """[(время начала, длительность, абсолютный url)], url init-сегмента (для fMP4)."""
    from urllib.parse import urljoin
    segs, t, dur, init = [], 0.0, None, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            try:
                dur = float(line[8:].split(",")[0])
            except ValueError:
                dur = 0.0
        elif line.startswith("#EXT-X-MAP:"):
            m = re.search(r'URI="([^"]+)"', line)
            if m:
                init = urljoin(url, m.group(1))
        elif line and not line.startswith("#") and dur is not None:
            segs.append((t, dur, urljoin(url, line)))
            t += dur
            dur = None
    return segs, init


def hls_fetch(url: str, a: float, b: float, base: Path, headers: dict | None = None,
              progress=None) -> tuple[Path, float]:
    """Скачивает сегменты HLS, покрывающие [a, b], в один «сырой» файл (без перепаковки).
    Возвращает (файл, время записи, с которого он начинается).

    Плейлист разбираем сами: так время каждого сегмента известно точно, а растущая запись идущего эфира
    (плейлист без #EXT-X-ENDLIST) не превращается в «прямую трансляцию», которую ffmpeg читал бы
    в реальном времени и не с того места."""
    from concurrent.futures import ThreadPoolExecutor
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", **(headers or {})})
    r = s.get(url, timeout=30)
    r.raise_for_status()
    text, purl = r.text, url
    if "#EXT-X-STREAM-INF" in text:  # пришёл master — берём первый вариант
        from urllib.parse import urljoin
        nxt = next(l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#"))
        purl = urljoin(url, nxt)
        r = s.get(purl, timeout=30)
        r.raise_for_status()
        text = r.text
    segs, init = _parse_media_playlist(text, purl)
    need = [sg for sg in segs if sg[0] + sg[1] > max(0.0, a) and sg[0] < b]
    if not need:
        raise TwitchCutError("в плейлисте нет сегментов для этого времени")

    def get(u: str) -> bytes:
        cands = [u]
        if "-unmuted" in u:
            cands.append(u.replace("-unmuted", "-muted"))
        last = None
        for cu in cands:
            for attempt in range(4):
                try:
                    rr = s.get(cu, timeout=(10, 40))
                    if rr.status_code == 200:
                        return rr.content
                    last = f"HTTP {rr.status_code}"
                    if rr.status_code in (403, 404):
                        break
                except requests.RequestException as e:
                    last = str(e)
                time.sleep(1 + attempt)
        raise TwitchCutError(f"сегмент не скачался: {last}")

    done = [0]

    def get_counted(u: str) -> bytes:
        data = get(u)
        done[0] += 1
        if progress:
            try:
                progress(done[0], len(need))
            except Exception:
                pass
        return data

    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(get_counted, [sg[2] for sg in need]))
    ext = ".mp4" if init else ".ts"
    tmp = base.with_name(base.name + ".part" + ext)
    with open(tmp, "wb") as fh:
        if init:
            fh.write(get(init))
        for p in parts:
            fh.write(p)
    return tmp, need[0][0]


def hls_download(url: str, a: float, b: float, base: Path, headers: dict | None = None) -> tuple[Path, float]:
    """Скачивает сегменты HLS, покрывающие [a, b]. Возвращает (файл, время записи, с которого он начинается)."""
    tmp, t0 = hls_fetch(url, a, b, base, headers)
    # Перепаковываем в MP4 (без перекодирования): по MPEG-TS ffmpeg перематывает неточно,
    # а у MP4 есть индекс кадров — перемотка точная до кадра.
    out = base.with_suffix(".mp4")
    tmp2 = base.with_name(base.name + ".remux.mp4")
    run([ffmpeg_bin(), "-y", "-v", "error", "-i", str(tmp), "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
         "-movflags", "+faststart", str(tmp2)])
    tmp.unlink(missing_ok=True)
    tmp2.replace(out)
    return out, t0


VARIANTS_DIAG: dict = {}   # последняя диагностика качеств (для интерфейса)


def twitch_vod_variants(vod_id: str) -> list[dict]:
    """Все качества VOD, включая 1440p/2160p (HEVC/AV1). Twitch отдаёт «источник» в 1440p только
    клиентам, которые заявляют поддержку HEVC/AV1; с токеном аккаунта шанс получить источник выше."""
    import random
    from .twitch_api import GQL, get_auth_token
    q = """query PlaybackAccessToken($vodID: ID!, $playerType: String!, $platform: String!) {
      videoPlaybackAccessToken(id: $vodID, params: {platform: $platform, playerBackend: "mediaplayer",
        playerType: $playerType}) { value signature }
    }"""
    token = get_auth_token()
    tries = [("аккаунт", token)] if token else []
    tries.append(("гость", None))
    diag = {"vod": vod_id, "attempts": [], "authed": bool(token)}
    best: list[dict] = []
    for who, auth in tries:
        try:
            d = GQL().query(q, {"vodID": str(vod_id), "playerType": "site", "platform": "web"},
                            prefer_web=True, auth=auth)
            tok = ((d.get("data") or {}).get("videoPlaybackAccessToken")) or {}
            if not tok.get("value"):
                raise TwitchCutError("нет токена воспроизведения")
        except Exception as e:
            diag["attempts"].append(f"{who}: токен — {e}")
            continue
        base = {"allow_source": "true", "allow_audio_only": "true", "allow_spectre": "true", "player": "twitchweb",
                "playlist_include_framerate": "true", "nauth": tok["value"], "nauthsig": tok["signature"],
                "p": str(random.randint(100000, 9999999))}
        for label, extra in (("hevc/av1", {"supported_codecs": "av1,h265,h264", "platform": "web",
                                           "player_backend": "mediaplayer", "include_unavailable": "true"}),
                             ("hevc", {"supported_codecs": "h265,h264"}),
                             ("стандарт", {})):
            try:
                r = requests.get(f"https://usher.ttvnw.net/vod/{vod_id}.m3u8", params={**base, **extra},
                                 timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    raise TwitchCutError(f"HTTP {r.status_code}: {r.text[:120]}")
                vs = parse_master_playlist(r.text)
                top = max((v["height"] for v in vs), default=0)
                diag["attempts"].append(f"{who}/{label}: {', '.join(v['name'] + ' ' + v['codec'] for v in vs if v['height'])}")
                if top > max((v["height"] for v in best), default=0):
                    best = vs
                if top >= 1440:
                    break
            except Exception as e:
                diag["attempts"].append(f"{who}/{label}: {e}")
        if max((v["height"] for v in best), default=0) >= 1440:
            break
    diag["best"] = max((f"{v['height']}p {v['codec']}" for v in best if v["height"]), default="", key=lambda x: int(x.split("p")[0]))
    VARIANTS_DIAG.clear()
    VARIANTS_DIAG.update(diag)
    log.info("Качества VOD %s: %s", vod_id, " | ".join(diag["attempts"]))
    if not best:
        raise TwitchCutError("; ".join(diag["attempts"]) or "плейлист недоступен")
    return best


class Source:
    def __init__(self, ref: str):
        ref = ref.strip().strip('"')
        self.ref = ref
        self.vod_id: Optional[str] = None
        self._info: Optional[dict] = None
        self._variants: Optional[list] = None
        self.last_quality = ""
        m = TWITCH_VOD_RE.search(ref)
        if m:
            self.kind = "twitch"
            self.vod_id = m.group(1)
        elif re.match(r"^\d{6,}$", ref):
            self.kind = "twitch"
            self.vod_id = ref
            self.ref = f"https://www.twitch.tv/videos/{ref}"
        elif ref.startswith(("http://", "https://")):
            self.kind = "url"
        else:
            p = Path(ref).expanduser()
            if not p.exists():
                raise TwitchCutError(f"Не понимаю источник: {ref!r}. Нужна ссылка на VOD Twitch или путь к файлу.")
            self.kind = "file"
            self.path = p.resolve()

    # ------------------------------------------------------------------ ids
    def job_id(self) -> str:
        if self.kind == "twitch":
            return f"vod_{self.vod_id}"
        if self.kind == "file":
            return "file_" + safe_name(self.path.stem, 40).replace(" ", "_")
        return "url_" + safe_name(re.sub(r"^https?://", "", self.ref), 40).replace("/", "_")

    # ------------------------------------------------------------- metadata
    def info(self) -> dict:
        if self._info is None:
            if self.kind == "file":
                self._info = {"title": self.path.stem, "duration": ffprobe_duration(self.path)}
            else:
                yt_dlp = _yt_dlp()
                opts = {"quiet": True, "no_warnings": True, "skip_download": True, "no_color": True,
                        **_cookie_opts()}
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        self._info = ydl.extract_info(self.ref, download=False)
                except Exception as e:  # yt_dlp.utils.DownloadError
                    msg = str(e)
                    if "subscriber" in msg.lower() or "sub-only" in msg.lower():
                        msg += "\nVOD только для подписчиков — такие пока не поддерживаются."
                    raise TwitchCutError(f"Не удалось получить информацию о видео: {msg}") from e
        return self._info

    def meta(self) -> dict:
        i = self.info()
        return {
            "kind": self.kind,
            "ref": self.ref,
            "vod_id": self.vod_id,
            "title": i.get("title") or "",
            "channel": i.get("uploader") or i.get("channel") or i.get("uploader_id") or "",
            "channel_id": i.get("uploader_id") or "",
            "duration": float(i.get("duration") or 0),
            "timestamp": i.get("timestamp"),
        }

    # ---------------------------------------------------------------- audio
    def fetch_audio(self, job_dir: Path, progress: ProgressFn = noop_progress) -> Path:
        """Скачивает/извлекает аудиодорожку целиком. Возвращает путь к файлу."""
        existing = _audio_files(job_dir)
        if existing:
            return existing[0]

        if self.kind == "file":
            out = job_dir / "audio_src.m4a"
            progress("audio", 0.1, "Извлекаю звук из файла")
            run([ffmpeg_bin(), "-y", "-v", "error", "-i", str(self.path), "-vn", "-ac", "1",
                 "-ar", "16000", "-c:a", "aac", "-b:a", "64k", str(out)])
            return out

        yt_dlp = _yt_dlp()

        def hook(d):
            if d.get("status") == "downloading":
                fi, fc = d.get("fragment_index"), d.get("fragment_count")
                if fi and fc:
                    frac = fi / fc
                else:
                    tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    frac = (d.get("downloaded_bytes", 0) / tot) if tot else 0
                progress("audio", min(frac, 0.99), f"Скачиваю звук стрима: {frac*100:.0f}%")

        opts = {
            # У Twitch есть отдельная аудиодорожка «Audio_Only» — она в ~20 раз меньше видео.
            "format": "Audio_Only/audio_only/bestaudio/worst",
            "outtmpl": str(job_dir / "audio_src.%(ext)s"),
            "quiet": True, "no_warnings": True, "noprogress": True, "no_color": True, **_cookie_opts(),
            "progress_hooks": [hook],
            "concurrent_fragment_downloads": 8,
            "retries": 10, "fragment_retries": 10,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([self.ref])
        except Exception as e:
            raise TwitchCutError(f"Не удалось скачать звук: {e}") from e
        files = _audio_files(job_dir)
        if not files:
            raise TwitchCutError("yt-dlp не сохранил аудиофайл")
        return files[0]

    # -------------------------------------------------------------- segment
    def _best_video_format(self, max_height: int) -> dict:
        """Лучшее качество не выше max_height. Для Twitch сначала спрашиваем плейлист напрямую с
        поддержкой HEVC/AV1 — только так Twitch отдаёт 1440p/4K (yt-dlp видит максимум 1080p)."""
        if self.kind == "twitch" and self._variants is None:
            try:
                self._variants = twitch_vod_variants(self.vod_id)
                log.info("Доступные качества VOD: %s",
                         ", ".join(f"{v['name']} {v['codec']}" for v in self._variants) or "нет")
            except Exception as e:
                log.warning("Не удалось получить расширенный список качеств (%s) — использую yt-dlp", e)
                self._variants = []
        video_vars = [v for v in (self._variants or []) if v["height"] > 0]
        if video_vars:
            ok = [v for v in video_vars if v["height"] <= max_height] or video_vars
            codec_rank = {"h264": 2, "hevc": 1, "av1": 0}
            ok.sort(key=lambda v: (v["height"], v["fps"], codec_rank.get(v["codec"], 0), v["bandwidth"]))
            return ok[-1]
        fmts = [f for f in (self.info().get("formats") or [])
                if f.get("vcodec") not in (None, "none") and f.get("url")]
        if not fmts:
            raise TwitchCutError("У видео не нашлось видеоформатов")
        ok = [f for f in fmts if (f.get("height") or 0) <= max_height] or fmts
        ok.sort(key=lambda f: ((f.get("height") or 0), (f.get("fps") or 0), (f.get("tbr") or 0)))
        f = ok[-1]
        return {"url": f["url"], "height": f.get("height") or 0, "fps": f.get("fps") or 0,
                "codec": (f.get("vcodec") or "")[:4], "name": f.get("format_id") or "", "bandwidth": f.get("tbr") or 0,
                "headers": f.get("http_headers") or {}}

    def refresh_variants(self) -> None:
        """Для растущей записи эфира плейлисты стоит перечитывать."""
        self._variants = None
        self._info = None

    def fetch_audio_range(self, a: float, b: float, out: Path, progress=None) -> Path:
        """Звук отрезка [a, b] (для прямого эфира) — без скачивания всей записи.

        Для Twitch сегменты качаются напрямую по плейлисту (параллельно, ~1 мин на 15 мин эфира), а время
        начала файла берётся из плейлиста — тем же способом, каким потом качается видео клипов. Поэтому звук
        для распознавания речи и видео клипа всегда совпадают до долей секунды."""
        stamp = int(time.time())  # своё имя у каждой попытки: старый файл может держать зависший процесс
        tmp = out.with_name(f"{out.stem}.part{stamp}{out.suffix}")
        if self.kind == "twitch":
            if self._variants is None:
                try:
                    self._variants = twitch_vod_variants(self.vod_id)
                except Exception as e:
                    log.warning("Плейлист Twitch: %s", e)
                    self._variants = []
            # только звук; если его нет — самое лёгкое видео (160p): там тот же звук
            vs = sorted(self._variants, key=lambda v: (v["codec"] != "audio", v["bandwidth"] or 1e12))
            last_err: Exception | None = None
            for v in vs[:2]:
                raw = None
                try:
                    raw, t0 = hls_fetch(v["url"], a, b, out.with_name(f"{out.stem}.dl{stamp}"),
                                       v.get("headers") or {}, progress)
                    # время отсчитываем ПОСЛЕ декодирования (-ss после -i): точно до сэмпла
                    run([ffmpeg_bin(), "-y", "-v", "error", "-i", str(raw), "-ss", f"{max(0.0, a - t0):.3f}",
                         "-t", f"{b - a:.3f}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k",
                         str(tmp)])
                    tmp.replace(out)
                    return out
                except Exception as e:
                    last_err = e
                    log.warning("Звук эфира через %s не скачался: %s", v.get("name") or v["codec"], e)
                finally:
                    if raw is not None:
                        raw.unlink(missing_ok=True)
            raise TwitchCutError(f"Не удалось скачать звук {fmt_time(a)}–{fmt_time(b)}: {last_err}")
        url, headers = (str(self.path) if self.kind == "file" else None), {}
        if not url:
            fmts = self.info().get("formats") or []
            af = [f for f in fmts if f.get("url") and (f.get("vcodec") in (None, "none") or "audio" in (f.get("format_id") or "").lower())]
            if not af:
                af = sorted([f for f in fmts if f.get("url")], key=lambda f: f.get("height") or 9999)
            if not af:
                raise TwitchCutError("Не найден поток со звуком")
            url, headers = af[0]["url"], af[0].get("http_headers") or {}
        cmd = [ffmpeg_bin(), "-y", "-v", "error"]
        if headers:
            cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
        cmd += ["-ss", f"{a:.3f}", "-i", url, "-t", f"{b - a:.3f}", "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "aac", "-b:a", "64k", str(tmp)]
        run(cmd)
        tmp.replace(out)
        return out

    def best_quality(self, max_height: int = 2160) -> str:
        try:
            v = self._best_video_format(max_height)
            return f"{v['height']}p{int(v['fps']) if v.get('fps') and v['fps'] > 31 else ''} {v.get('codec', '')}".strip()
        except Exception:
            return ""

    def fetch_segment(self, start: float, end: float, out: Path, max_height: int = 2160) -> tuple[Path, float]:
        """Готовит видеофрагмент [start, end]. Возвращает (путь, точное смещение начала клипа внутри файла).

        HLS-сегменты скачиваются напрямую по плейлисту: так мы точно знаем, с какой секунды записи начинается
        файл, и субтитры/заглушение мата совпадают с речью до кадра. Для локального файла ничего не копируем.
        """
        if self.kind == "file":
            return self.path, start
        base = out.with_suffix("")
        meta_p = base.with_suffix(".json")
        info = read_json(meta_p)
        f = self._best_video_format(max_height)
        best_q = f"{f['height']}p{int(f['fps']) if f.get('fps') and f['fps'] > 31 else ''} {f.get('codec', '')}".strip()
        if info:
            cached = Path(info["file"])
            same_q = int(str(info.get("quality", "0")).split("p")[0] or 0) >= f["height"]
            if cached.exists() and cached.stat().st_size > 10_000 and info.get("start", 1e18) <= start \
                    and info.get("end", -1) >= end and same_q:
                self.last_quality = info.get("quality", "")
                return cached, start - info["t0"]
        for old in (base.with_suffix(".mkv"), base.with_suffix(".mp4"), base.with_suffix(".ts")):
            old.unlink(missing_ok=True)  # старые фрагменты могли быть со сдвигом — не используем
        self.last_quality = best_q
        log.info("Скачиваю фрагмент %s–%s (%s)", fmt_time(start), fmt_time(end), self.last_quality)
        try:
            path, t0 = hls_download(f["url"], start - 1.0, end + 1.0, base, f.get("headers") or {})
        except Exception as e:
            log.warning("Прямая загрузка сегментов не удалась (%s) — запасной способ через ffmpeg", e)
            path, t0 = self._ffmpeg_segment(f, start, end, base.with_suffix(".mp4"))
        write_json(meta_p, {"file": str(path), "t0": t0, "start": start, "end": end, "quality": self.last_quality})
        return path, start - t0

    def _ffmpeg_segment(self, f: dict, start: float, end: float, out: Path) -> tuple[Path, float]:
        """Запасной путь: ffmpeg + MP4 (список правок MP4 сохраняет точное начало)."""
        margin = 2.0 if start >= 2.0 else start
        cmd = [ffmpeg_bin(), "-y", "-v", "error"]
        headers = f.get("headers") or {}
        if headers:
            cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
        tmp = out.with_name(out.stem + ".part.mp4")
        cmd += ["-ss", f"{start - margin:.3f}", "-i", f["url"], "-t", f"{end - start + margin + 1.0:.3f}",
                "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", str(tmp)]
        run(cmd)
        tmp.replace(out)
        return out, start - margin
        f = self._best_video_format(max_height)
        self.last_quality = f"{f['height']}p {f.get('codec', '')}".strip()
        cmd = [ffmpeg_bin(), "-y", "-v", "error"]
        headers = f.get("headers") or {}
        if headers:
            cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
        tmp = out.with_name(out.stem + ".part" + out.suffix)
        cmd += ["-ss", f"{start - margin:.3f}", "-i", f["url"], "-t", f"{end - start + margin + 1.0:.3f}",
                "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", str(tmp)]
        log.info("Скачиваю фрагмент %s (%s)", out.name, self.last_quality)
        run(cmd)
        tmp.replace(out)
        return out, margin

    # --------------------------------------------------------- viewer clips
    def viewer_clips(self, client_id: str, client_secret: str) -> list[dict]:
        """Клипы, которые нарезали зрители этого VOD (Twitch Helix API). Нужен client_id/secret."""
        if self.kind != "twitch" or not client_id or not client_secret:
            return []
        try:
            tok = requests.post("https://id.twitch.tv/oauth2/token", params={
                "client_id": client_id, "client_secret": client_secret,
                "grant_type": "client_credentials"}, timeout=20).json()["access_token"]
            h = {"Client-Id": client_id, "Authorization": f"Bearer {tok}"}
            v = requests.get("https://api.twitch.tv/helix/videos", params={"id": self.vod_id},
                             headers=h, timeout=20).json()["data"][0]
            created = dt.datetime.fromisoformat(v["created_at"].replace("Z", "+00:00"))
            started = created - dt.timedelta(minutes=5)
            ended = created + dt.timedelta(days=3)
            out, cursor = [], None
            for _ in range(20):
                params = {"broadcaster_id": v["user_id"], "first": 100,
                          "started_at": started.isoformat().replace("+00:00", "Z"),
                          "ended_at": ended.isoformat().replace("+00:00", "Z")}
                if cursor:
                    params["after"] = cursor
                r = requests.get("https://api.twitch.tv/helix/clips", params=params, headers=h, timeout=20).json()
                for c in r.get("data", []):
                    if c.get("video_id") == self.vod_id and c.get("vod_offset") is not None:
                        out.append({"offset": float(c["vod_offset"]), "duration": float(c.get("duration") or 30),
                                    "views": int(c.get("view_count") or 0), "title": c.get("title") or ""})
                cursor = (r.get("pagination") or {}).get("cursor")
                if not cursor:
                    break
            log.info("Клипов от зрителей найдено: %d", len(out))
            return out
        except Exception as e:
            log.warning("Не удалось получить клипы зрителей: %s", e)
            return []
