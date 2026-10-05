"""Локальный веб-интерфейс (только стандартная библиотека Python).

Задачи выполняются по одной в фоновом потоке — так сервис не перегружает слабый ПК/VPS.
"""
from __future__ import annotations

import json
import mimetypes
import queue
import re
import shutil
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import streamers
from .config import load_config
from .twitch_api import channel_info, channel_vods
from .pipeline import STAGE_TITLES, STAGES, Job
from .util import ROOT_DIR, TwitchCutError, log, read_json, write_json

STATIC = Path(__file__).resolve().parent / "static"


class Manager:
    def __init__(self, config_path: Path | None):
        self.config_path = config_path
        self.base_cfg = load_config(config_path)
        self.ws = Path(self.base_cfg["workspace"])
        self.ws.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue = queue.Queue()
        self.live: dict[str, dict] = {}          # id -> state (в памяти, для текущих задач)
        self.manual_events: dict[str, threading.Event] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._fill_avatars, daemon=True).start()
        self.rendering: dict[str, dict] = {}       # "job/n" -> перемонтаж клипа
        self.compiling: dict[str, dict] = {}       # job -> склейка «Топ-N»
        self.port = 8765
        self.render_errors: dict[str, str] = {}
        self._cleanup_stale()
        self.live_jobs: dict[str, tuple] = {}      # vod-job id -> (LiveJob, Thread)
        self.live_status: dict[str, dict] = {}     # login -> статус эфира
        threading.Thread(target=self._watcher, daemon=True).start()
        from . import cleanup
        cleanup.start_background(lambda: self.ws,
                                 lambda: load_config(self.config_path).get("cleanup", {}).get("keep_days", 2),
                                 lambda: set(self.live_jobs))

    # ------------------------------------------------------------ live
    def _start_live(self, vod_id: str, login: str) -> None:
        from .live import LiveJob
        jid = f"live_{vod_id}"
        cur = self.live_jobs.get(jid)
        if cur and cur[1].is_alive():
            return
        cfg = load_config(self.config_path)

        def on_update(st, _id=jid):
            self.live[_id] = dict(st)

        job = LiveJob(vod_id, login, cfg, on_update=on_update)

        def target():
            try:
                job.run_live()
            except Exception:
                log.error("Эфир %s упал:\n%s", login, traceback.format_exc())
            finally:
                self.live.pop(jid, None)

        t = threading.Thread(target=target, daemon=True, name=f"live-{login}")
        self.live_jobs[jid] = (job, t)
        t.start()
        log.info("Начинаю следить за эфиром %s (запись %s)", login, vod_id)

    def _watcher(self) -> None:
        """Раз в пару минут проверяет отслеживаемых стримеров: начался ли эфир."""
        from .twitch_api import live_status
        time.sleep(3)
        while True:
            cfg = load_config(self.config_path)
            for st in streamers.load_all():
                if not st.get("watch"):
                    self.live_status.pop(st["login"], None)
                    # слежку выключили — останавливаем и уже идущую обработку эфира
                    for jid, (job, t) in list(self.live_jobs.items()):
                        if job.login == st["login"] and t.is_alive():
                            log.info("Слежка за %s выключена — останавливаю обработку эфира", st["login"])
                            job.stop_event.set()
                    continue
                try:
                    ls = live_status(st["login"])
                except Exception as e:
                    self.live_status[st["login"]] = {"live": False, "error": str(e), "checked": time.time()}
                    continue
                ls["checked"] = time.time()
                self.live_status[st["login"]] = ls
                if ls["live"] and ls.get("vod"):
                    jid = f"live_{ls['vod']['id']}"
                    ls["job"] = jid
                    j = read_json(self.ws / jid / "job.json") or {}
                    if j.get("live_stopped"):
                        ls["stopped"] = True  # пользователь остановил наблюдение за этим эфиром
                    else:
                        self._start_live(str(ls["vod"]["id"]), st["login"])
                elif ls["live"]:
                    ls["error"] = "Канал в эфире, но запись (VOD) не ведётся — анализ эфира невозможен"
                else:
                    # эфир закончился, пока программа была закрыта, — дообработать хвост
                    for d in self.ws.glob("live_*"):
                        j = read_json(d / "job.json") or {}
                        if (j.get("status") == "live" and j.get("streamer") == st["login"] and not j.get("live_stopped")
                                and d.name not in self.live_jobs):
                            self._start_live(d.name[5:], st["login"])
            time.sleep(max(30, int(cfg["live"]["poll_seconds"])))

    def stop_live(self, job_id: str) -> None:
        cur = self.live_jobs.get(job_id)
        if cur and cur[1].is_alive():
            cur[0].update(live_stopped=True, message="Останавливаю после текущего шага…")
            cur[0].stop_event.set()
            return
        st = read_json(self.ws / job_id / "job.json") or {}
        if st:
            clips = read_json(self.ws / job_id / "clips.json") or []
            st.update(status="done", live=False, live_stopped=True, message=f"Наблюдение остановлено. Клипов: {len(clips)}")
            write_json(self.ws / job_id / "job.json", st)
        self.live.pop(job_id, None)

    def _cleanup_stale(self) -> None:
        """После перезапуска программы прерванные задачи автоматически продолжаются с места остановки
        (всё, что уже скачано, распознано и оценено, сохранено по кускам)."""
        streams_watch = {s["login"] for s in streamers.load_all() if s.get("watch")}
        resume = []
        for d in self.ws.iterdir():
            jp = d / "job.json"
            st = read_json(jp) if jp.exists() else None
            if not st:
                continue
            status = st.get("status")
            if status in ("running", "queued", "awaiting_llm") and st.get("ref"):
                resume.append((st.get("updated", 0), d.name, st))
            elif status == "live" and st.get("live_stopped"):
                # наблюдение остановили, а программу перезапустили до конца шага — задача завершена
                clips = read_json(d / "clips.json") or []
                st.update(status="done", live=False, message=f"Наблюдение остановлено. Клипов: {len(clips)}")
                write_json(jp, st)
            elif status == "live" and st.get("streamer") not in streams_watch:
                clips = read_json(d / "clips.json") or []
                st.update(status="done", live=False, live_stopped=True,
                          message=f"Наблюдение за эфиром выключено. Клипов: {len(clips)}")
                write_json(jp, st)
        for _, jid, st in sorted(resume):
            log.info("Продолжаю прерванную задачу %s", jid)
            try:
                st.update(message="Продолжаю после перезапуска…")
                write_json(self.ws / jid / "job.json", st)
                self.submit(st["ref"], st.get("options") or {})
            except Exception as e:
                log.warning("Не удалось продолжить %s: %s", jid, e)

    # --------------------------------------------------------- one clip
    def rerender_clip(self, job_id: str, n: int, speed: float, trim: bool, layout: str | None = None,
                      action: str = "render", target: float | None = None) -> None:
        key = f"{job_id}/{n}"
        if key in self.rendering:
            raise TwitchCutError("Этот клип уже перемонтируется")
        st = read_json(self.ws / job_id / "job.json") or {}
        if not st.get("ref"):
            raise TwitchCutError("Задача не найдена")
        self.rendering[key] = {"speed": speed, "trim": trim, "status": "waiting"}

        def target():
            from .pipeline import Job, heavy
            try:
                cfg = load_config(self.config_path, options_to_overrides(st.get("options") or {}))
                job = Job(st["ref"], cfg, options=st.get("options") or {})
                with heavy(f"перемонтаж клипа {n}"):
                    self.rendering[key]["status"] = "shortening" if action == "shorten" else "rendering"
                    if action == "shorten":
                        job.shorten_clip(n, float(target or 30), speed, trim, layout)
                    elif action == "restore":
                        job.restore_clip(n, speed, trim, layout)
                    else:
                        job.rerender_clip(n, speed, trim, layout)
            except Exception as e:
                log.error("Перемонтаж клипа %s: %s", key, e)
                self.render_errors[key] = str(e)
            finally:
                self.rendering.pop(key, None)

        self.render_errors.pop(key, None)
        threading.Thread(target=target, daemon=True).start()

    def make_compilation(self, job_id: str, n: int) -> None:
        """Склейка «Топ-N» в фоне (по очереди с остальной тяжёлой работой)."""
        if (self.compiling.get(job_id) or {}).get("status") == "running":
            raise TwitchCutError("Склейка уже собирается")
        d = self.ws / job_id
        clips = read_json(d / "clips.json") or []
        if not clips:
            raise TwitchCutError("У задачи нет готовых клипов")
        self.compiling[job_id] = {"status": "running", "n": n}
        st = read_json(d / "job.json") or {}

        def work():
            from .compilation import build_top
            from .pipeline import heavy
            from . import feedback
            try:
                cfg = load_config(self.config_path, options_to_overrides(st.get("options") or {}))
                with heavy(f"склейка «Топ-{n}»"):
                    build_top(d, clips, feedback.for_job(job_id), cfg, n)
                self.compiling.pop(job_id, None)
            except Exception as e:
                log.error("Склейка %s: %s", job_id, e)
                self.compiling[job_id] = {"status": "error", "error": str(e)}
        threading.Thread(target=work, daemon=True).start()

    def best_build(self, data: dict) -> None:
        """Подборка «Лучшее» из архива моментов — в фоне."""
        if (self.compiling.get("__best__") or {}).get("status") == "running":
            raise TwitchCutError("Подборка уже собирается")
        self.compiling["__best__"] = {"status": "running"}

        def work():
            from . import highlights
            from .pipeline import heavy
            try:
                cfg = load_config(self.config_path)
                with heavy("подборка «Лучшее»"):
                    highlights.build(cfg, data.get("streamer") or None, float(data["days"]) if data.get("days") else None,
                                     int(data.get("n") or 5), float(data.get("max_total") or 60), data.get("ids") or None)
                self.compiling.pop("__best__", None)
            except Exception as e:
                log.error("Подборка «Лучшее»: %s", e)
                self.compiling["__best__"] = {"status": "error", "error": str(e)}
        threading.Thread(target=work, daemon=True).start()

    def best_extract(self, job_id: str) -> None:
        """Выбрать лучшие фразы из уже готовых клипов задачи (для старых задач) — в фоне."""
        d = self.ws / job_id
        clips = [c for c in (read_json(d / "clips.json") or []) if (d / c["file"]).exists()]
        if not clips:
            raise TwitchCutError("У задачи нет готовых клипов (возможно, удалены автоочисткой)")
        st = read_json(d / "job.json") or {}
        key = "__extract__" + job_id
        self.compiling[key] = {"status": "running"}

        def work():
            from . import highlights
            from .pipeline import heavy
            try:
                cfg = load_config(self.config_path)
                with heavy("поиск лучших фраз"):
                    added = highlights.extract(d, clips, cfg, {}, streamer=st.get("streamer") or "")
                self.compiling[key] = {"status": "done", "added": len(added)}
            except Exception as e:
                log.error("Лучшие фразы %s: %s", job_id, e)
                self.compiling[key] = {"status": "error", "error": str(e)}
        threading.Thread(target=work, daemon=True).start()

    def send_to_tiktok(self, job_id: str, n: int | None = None, file: str | None = None,
                       account: str | None = None) -> dict:
        """Загрузить клип (или склейку) во «Входящие» TikTok как черновик."""
        from . import tiktok_upload
        from .compilation import _duration
        d = self.ws / job_id
        st = read_json(d / "job.json") or {}
        streamer = st.get("streamer") or (st.get("channel") or "").lower()
        acc = account or tiktok_upload.account_for(streamer)
        if n is not None:
            clips = read_json(d / "clips.json") or []
            c = next((x for x in clips if x.get("n") == n), None)
            if not c:
                raise TwitchCutError("Клип не найден")
            res = tiktok_upload.upload_draft(d / c["file"], acc)
            tiktok_upload.register(res, acc, "clip", job_id, c["file"], _duration(d / c["file"]), streamer)
            clips = read_json(d / "clips.json") or clips
            for x in clips:
                if x.get("n") == n:
                    x["tiktok_draft"] = res
            write_json(d / "clips.json", clips)
            return res
        comps = read_json(d / "compilations.json") or []
        c = next((x for x in comps if x.get("file") == file), None)
        if not c:
            raise TwitchCutError("Склейка не найдена")
        res = tiktok_upload.upload_draft(d / c["file"], acc)
        tiktok_upload.register(res, acc, "compilation", job_id, c["file"], _duration(d / c["file"]), streamer)
        c["tiktok_draft"] = res
        write_json(d / "compilations.json", comps)
        return res

    def on_tiktok_published(self, up: dict, url: str, stats: dict) -> None:
        """Черновик опубликован: ставим ссылку, отмечаем «Выложил», подтягиваем просмотры."""
        from . import feedback, tiktok
        kind, job, file = up.get("kind"), up.get("job") or "", up.get("file") or ""
        if kind == "clip":
            prev = next((x for x in feedback.load() if x.get("id") == f"{job}/{file}"), {})
            if prev.get("status") == "posted" and tiktok.is_tiktok(prev.get("url") or ""):
                rec = {**prev}  # ссылку уже вставили руками — не трогаем
            else:
                try:
                    rec = self.save_feedback({"job": job, "file": file, "status": "posted", "url": url,
                                              "title": prev.get("title"), "views": prev.get("views"),
                                              "likes": prev.get("likes")}, fetch=False)
                except TwitchCutError:  # задачу уже удалили — запишем отметку по данным черновика
                    rec = feedback.upsert({"job": job, "file": file, "streamer": up.get("streamer"),
                                           "status": "posted", "url": url})
            rec = {k: v for k, v in rec.items() if k != "stats_pending"}
            rec.update(tt_open_id=up.get("open_id"), video_id=up.get("video_id"), auto_link=True)
            if stats:
                rec = tiktok.apply(rec, stats)
            feedback.replace(rec)
            if not stats:
                feedback.fetch_stats_async(rec)
            clips = read_json(self.ws / job / "clips.json")
            if clips:
                for x in clips:
                    if x.get("file") == file and x.get("tiktok_draft"):
                        x["tiktok_draft"] = {**x["tiktok_draft"], "url": url}
                write_json(self.ws / job / "clips.json", clips)
            return
        if kind == "compilation":
            path = self.ws / job / "compilations.json"
        else:
            from . import highlights
            path = highlights.ARCHIVE / "compilations.json"
        comps = read_json(path)
        if comps:
            for x in comps:
                if x.get("file") == file:
                    x["tiktok_draft"] = {**(x.get("tiktok_draft") or {}), "url": url}
                    if stats.get("views") is not None:
                        x["tiktok_views"] = stats["views"]
            write_json(path, comps)

    def _fill_avatars(self) -> None:
        """Подтягивает аватарки стримеров, добавленных без доступа к Twitch."""
        for st in streamers.load_all():
            if not st.get("avatar"):
                try:
                    streamers.upsert({"login": st["login"]})
                except Exception as e:
                    log.debug("Аватар %s: %s", st["login"], e)

    # ------------------------------------------------------------ jobs
    def list_jobs(self) -> list[dict]:
        out = []
        for d in sorted(self.ws.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir():
                st = self.live.get(d.name) or read_json(d / "job.json")
                if st:
                    item = {k: st.get(k) for k in ("id", "title", "channel", "status", "stage", "message",
                                                   "stage_progress", "created", "duration", "streamer_avatar")}
                    clips = read_json(d / "clips.json")
                    item["clips"] = len(clips) if clips else 0
                    out.append(item)
        return out

    def get(self, job_id: str) -> dict | None:
        d = self.ws / job_id
        st = dict(self.live.get(job_id) or read_json(d / "job.json") or {})
        if not st:
            return None
        if st.get("status") in ("done", "error") or not st.get("clips"):
            clips = read_json(d / "clips.json")
            if clips is None:
                clips = read_json(d / "clips.partial.json")
            if clips is not None:
                st["clips"] = clips
        if st.get("clips"):
            from . import feedback
            fb = feedback.for_job(job_id)
            st["clips"] = [{**c, "feedback": fb.get(c["file"]), "missing": not (d / c["file"]).exists()}
                           for c in st["clips"]]
        st["rendering"] = {k.split("/", 1)[1]: v["status"] for k, v in self.rendering.items() if k.startswith(job_id + "/")}
        st["render_errors"] = {k.split("/", 1)[1]: v for k, v in self.render_errors.items() if k.startswith(job_id + "/")}
        st["timeline"] = read_json(d / "timeline.json")
        st["compilations"] = [x for x in (read_json(d / "compilations.json") or []) if (d / x["file"]).exists()]
        st["compiling"] = self.compiling.get(job_id)
        cands = read_json(d / "all_candidates.json") or read_json(d / "candidates.json") or []
        rank = (read_json(d / "ranking.json") or {}).get("ranking", {})
        st["candidates"] = [{"id": c["id"], "peak": c["peak"], "signal": c.get("signal", 0), "source": c.get("source"),
                             "topic": c.get("topic"),
                             "score": (rank.get(c["id"]) or {}).get("score"),
                             "reason": (rank.get(c["id"]) or {}).get("reason"),
                             "reaction": c.get("reaction")} for c in cands]
        if st.get("status") == "awaiting_llm":
            pp = d / "llm_prompt.txt"
            st["llm_prompt"] = pp.read_text(encoding="utf-8") if pp.exists() else ""
        st["stages"] = [{"key": s, "title": STAGE_TITLES[s]} for s in STAGES]
        return st

    def submit(self, ref: str, options: dict, force_from: str | None = None) -> str:
        from .sources import Source
        src = Source(ref)  # валидация + id
        job_id = src.job_id()
        d = self.ws / job_id
        d.mkdir(parents=True, exist_ok=True)
        st = read_json(d / "job.json") or {"id": job_id, "created": time.time()}
        if st.get("status") in ("running", "queued", "awaiting_llm", "live") and job_id in self.live:
            raise TwitchCutError("Эта задача уже выполняется")
        st.update(ref=ref, status="queued", message="В очереди", options=options, error=None)
        write_json(d / "job.json", st)
        self.live[job_id] = st
        self.q.put((job_id, ref, options, force_from))
        return job_id

    def delete_job(self, job_id: str) -> None:
        if job_id in self.live and (self.live[job_id].get("status") in ("running", "queued", "awaiting_llm", "live")):
            raise TwitchCutError("Задача ещё выполняется")
        d = (self.ws / job_id).resolve()
        if d.parent != self.ws.resolve() or not d.is_dir():
            raise TwitchCutError("Задача не найдена")
        shutil.rmtree(d, ignore_errors=True)

    def status(self) -> dict:
        from .llm import claude_status
        from .util import ffmpeg_bin
        cfg = load_config(self.config_path)
        st: dict = {"llm_backend": cfg["llm"]["backend"], "cli_model": cfg["llm"].get("cli_model"),
                    "api_model": cfg["llm"].get("model")}
        try:
            ffmpeg_bin()
            st["ffmpeg"] = True
        except TwitchCutError:
            st["ffmpeg"] = False
        try:
            import faster_whisper  # noqa
            st["whisper"] = True
        except ImportError:
            st["whisper"] = False
        from .transcribe import wcpp_status
        st["gpu_asr"] = wcpp_status()
        st["whisper_backend"] = cfg["whisper"].get("backend", "auto")
        if st["gpu_asr"]["installed"] and st["whisper_backend"] != "faster-whisper":
            st["whisper"] = True  # распознаёт видеокарта, даже если faster-whisper не установлен
        try:
            import yt_dlp  # noqa
            st["ytdlp"] = True
        except ImportError:
            st["ytdlp"] = False
        st["claude"] = claude_status()
        from .cleanup import disk_usage
        try:
            st["disk"] = {**disk_usage(self.ws), "keep_days": cfg.get("cleanup", {}).get("keep_days", 2)}
        except OSError:
            st["disk"] = None
        st["queue"] = self.q.qsize()
        st["live"] = self.live_status
        from .pipeline import HEAVY_STATE
        st["busy"] = HEAVY_STATE.get("owner") or ""
        return st

    def streamer_snapshot(self, login: str) -> Path:
        """Кадр из последнего скачанного видео стримера — чтобы показать/указать вебку мышкой."""
        from .util import ffmpeg_bin, run
        login = login.lower()
        segs = []
        for d in self.ws.iterdir():
            st = read_json(d / "job.json") or {}
            if st.get("streamer") == login and (d / "segments").is_dir():
                segs += [p for p in (d / "segments").glob("*.mp4") if p.stat().st_size > 1_000_000]
        if not segs:
            raise TwitchCutError("Пока нет скачанного видео этого стримера — сначала нарежьте хотя бы одну запись")
        seg = max(segs, key=lambda p: p.stat().st_mtime)
        out = ROOT_DIR / "data" / "snapshots" / f"{login}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or out.stat().st_mtime < seg.stat().st_mtime:
            run([ffmpeg_bin(), "-y", "-v", "error", "-ss", "3", "-i", str(seg), "-frames:v", "1", "-vf", "scale=960:-2",
                 "-q:v", "4", str(out)])
        return out

    def save_feedback(self, data: dict, fetch: bool = True) -> dict:
        from . import feedback
        job_id, file = data.get("job") or "", data.get("file") or ""
        d = self.ws / job_id
        clips = read_json(d / "clips.json") or []
        c = next((x for x in clips if x.get("file") == file), None)
        if not c:
            raise TwitchCutError("Клип не найден")
        st = read_json(d / "job.json") or {}
        text = c.get("text") or ""
        if not text:
            tr = read_json(d / "transcript.json") or {}
            text = " ".join(w["w"] for w in tr.get("words", []) if c["start"] <= w["s"] <= c["end"])[:600]

        def num(v):
            try:
                return int(str(v).replace(" ", "").replace(",", "")) if str(v).strip() not in ("", "None") else None
            except ValueError:
                return None
        entry = {
            "job": job_id, "file": file, "streamer": st.get("streamer") or (st.get("channel") or "").lower(),
            "title": data.get("title") or c.get("title"), "titles": c.get("titles"), "category": c.get("category"),
            "source": c.get("source"), "topic": c.get("topic"), "score": c.get("score"), "duration": c.get("duration"),
            "text": text, "status": data.get("status"), "reasons": data.get("reasons") or [],
            "note": (data.get("note") or "").strip()[:300], "views": num(data.get("views")), "likes": num(data.get("likes")),
            "url": (data.get("url") or "").strip()[:300], "final_duration": c.get("final_duration"),
            "speed": c.get("speed"), "layout": c.get("layout"), "extra": c.get("extra"),
        }
        prev = next((x for x in feedback.load() if x.get("id") == f"{job_id}/{file}"), {})
        rec = feedback.upsert(entry)
        if fetch and rec.get("url") and (rec.get("url") != prev.get("url") or not rec.get("stats_at") or data.get("refresh")):
            feedback.fetch_stats_async(rec)
            rec = {**rec, "stats_pending": True}
        return rec

    def manual_answer(self, job_id: str, text: str) -> None:
        d = self.ws / job_id
        (d / "llm_response.txt").write_text(text, encoding="utf-8")
        ev = self.manual_events.get(job_id)
        if ev:
            ev.set()

    # ---------------------------------------------------------- worker
    def _worker(self) -> None:
        while True:
            job_id, ref, options, force_from = self.q.get()
            try:
                cfg = load_config(self.config_path, options_to_overrides(options))

                def on_update(st, _id=job_id):
                    self.live[_id] = dict(st)

                def wait_manual(prompt_path: Path, resp_path: Path, _id=job_id):
                    ev = threading.Event()
                    self.manual_events[_id] = ev
                    while not resp_path.exists():
                        ev.wait(2)
                    self.manual_events.pop(_id, None)

                job = Job(ref, cfg, chat_file=options.get("chat_file") or None,
                          on_update=on_update, wait_manual=wait_manual, options=options)
                job.state["options"] = options
                job.run(force_from=force_from)
            except TwitchCutError as e:
                log.error("Задача %s: %s", job_id, e)
            except Exception:
                log.error("Задача %s упала:\n%s", job_id, traceback.format_exc())
            finally:
                time.sleep(0.5)
                self.live.pop(job_id, None)


def options_to_overrides(o: dict) -> dict:
    ov: dict = {}
    if o.get("max_clips"):
        ov.setdefault("clips", {})["max_clips"] = int(o["max_clips"])
    if o.get("llm"):
        ov.setdefault("llm", {})["backend"] = o["llm"]
    if o.get("analysis_mode") in ("deep", "fast"):
        ov.setdefault("analysis", {})["mode"] = o["analysis_mode"]
    if o.get("cli_model"):
        ov.setdefault("llm", {})["cli_model"] = o["cli_model"]
    if o.get("layout"):
        ov.setdefault("render", {})["layout"] = o["layout"]
    if "censor" in o:
        ov.setdefault("censor", {})["enabled"] = bool(o["censor"])
    if o.get("streamer_context"):
        ov.setdefault("llm", {})["streamer_context"] = o["streamer_context"]
    return ov


def make_handler(mgr: Manager):
    class H(BaseHTTPRequestHandler):
        server_version = "TwitchCut"

        def log_message(self, fmt, *args):  # тише в консоли
            pass

        # ------------------------------------------------------- helpers
        def _json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                return json.loads(raw.decode("utf-8") or "{}")
            except ValueError:
                return {}

        def _file(self, path: Path):
            if not path.is_file():
                return self._json({"error": "not found"}, 404)
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".txt":
                ctype = "text/plain; charset=utf-8"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                else:
                    start = max(0, size - int(m.group(2)))
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if "download" in (urlparse(self.path).query or ""):
                from urllib.parse import quote
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name)}")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # браузер сам оборвал загрузку (перемотка видео, закрыта вкладка) — это нормально

        # -------------------------------------------------------- routes
        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if path.startswith("/static/"):
                target = (STATIC / path[len("/static/"):]).resolve()
                if STATIC.resolve() not in target.parents:
                    return self._json({"error": "forbidden"}, 403)
                return self._file(target)
            if path in ("/tiktok/callback", "/tiktok/callback/"):
                from urllib.parse import parse_qs
                from . import tiktok_upload
                q = parse_qs(urlparse(self.path).query)
                try:
                    if q.get("error"):
                        raise TwitchCutError("TikTok: " + (q.get("error_description") or q["error"])[0])
                    tok = tiktok_upload.finish_login(q.get("code", [""])[0], q.get("state", [""])[0])
                    msg = (f"Готово: аккаунт TikTok «{tok.get('display_name') or 'аккаунт'}» подключён"
                           + (f" и привязан к стримеру {tok['streamer']}" if tok.get("streamer") else "")
                           + ". Вкладку можно закрыть.")
                except Exception as e:
                    msg = f"Не удалось войти в TikTok: {e}"
                body = (f"<!doctype html><meta charset=utf-8><title>TwitchCut</title><body style='font:16px sans-serif;"
                        f"background:#0e0e10;color:#efeff1;padding:40px'><h2>{msg}</h2>"
                        f"<p><a style='color:#a970ff' href='/#/status'>Вернуться в TwitchCut</a></p>").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                if path == "/api/tiktok/status":
                    from . import tiktok_upload
                    return self._json(tiktok_upload.status())
                if path == "/api/tiktok/login":
                    from . import tiktok_upload
                    from urllib.parse import parse_qs
                    who = parse_qs(urlparse(self.path).query).get("streamer", [""])[0]
                    url = tiktok_upload.login_url(mgr.port, who or None)
                    self.send_response(302)
                    self.send_header("Location", url)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path == "/api/highlights":
                    from urllib.parse import parse_qs
                    from . import highlights
                    q = parse_qs(urlparse(self.path).query)
                    who, days = q.get("streamer", [""])[0], float(q.get("days", ["0"])[0] or 0)
                    items = highlights.rank(highlights.load())
                    if who:
                        items = [m for m in items if m.get("streamer") == who]
                    if days:
                        items = [m for m in items if time.time() - m.get("created", 0) <= days * 86400]
                    comps = [c for c in (read_json(highlights.ARCHIVE / "compilations.json") or [])
                             if (ROOT_DIR / c["file"]).exists()]
                    extracting = {k[len("__extract__"):]: v for k, v in mgr.compiling.items() if k.startswith("__extract__")}
                    return self._json({"items": items[:300], "total": len(items), "size": highlights.size_bytes(),
                                       "compilations": comps[::-1], "building": mgr.compiling.get("__best__"),
                                       "extracting": extracting})
                if path == "/api/status":
                    return self._json(mgr.status())
                if path == "/api/feedback":
                    from . import feedback
                    from urllib.parse import parse_qs
                    login = parse_qs(urlparse(self.path).query).get("streamer", [""])[0] or None
                    items = [f for f in feedback.load() if not login or f.get("streamer") == login]
                    return self._json({"items": sorted(items, key=lambda f: -f.get("updated", 0)),
                                       "stats": feedback.stats(login), "reasons": feedback.REASONS,
                                       "by_streamer": {} if login else feedback.stats_by_streamer()})
                if path == "/api/twitch/auth":
                    from .twitch_api import auth_info
                    return self._json(auth_info())
                if path == "/api/twitch/qualities":
                    from urllib.parse import parse_qs
                    from .sources import TWITCH_VOD_RE, VARIANTS_DIAG, twitch_vod_variants
                    q = parse_qs(urlparse(self.path).query).get("vod", [""])[0]
                    m = TWITCH_VOD_RE.search(q)
                    vid = m.group(1) if m else re.sub(r"\D", "", q)
                    if not vid:
                        raise TwitchCutError("Укажите ссылку на VOD")
                    try:
                        vs = twitch_vod_variants(vid)
                    except TwitchCutError:
                        vs = []
                    return self._json({"variants": [{k: v[k] for k in ("name", "height", "fps", "codec")} for v in vs
                                                    if v["height"]], "diag": dict(VARIANTS_DIAG)})
                if path == "/api/streamers":
                    items = streamers.load_all()
                    for it in items:
                        it["live_status"] = mgr.live_status.get(it["login"])
                    return self._json(items)
                m = re.match(r"^/api/streamers/([\w]+)/snapshot$", path)
                if m:
                    return self._file(mgr.streamer_snapshot(m.group(1)))
                m = re.match(r"^/api/streamers/([\w]+)/vods$", path)
                if m:
                    return self._json(channel_vods(m.group(1)))
                m = re.match(r"^/api/channel/([\w]+)$", path)
                if m:
                    return self._json(channel_info(m.group(1)))
            except TwitchCutError as e:
                return self._json({"error": str(e)}, 400)
            if path == "/api/jobs":
                return self._json(mgr.list_jobs())
            m = re.match(r"^/api/jobs/([\w.-]+)$", path)
            if m:
                st = mgr.get(m.group(1))
                return self._json(st) if st else self._json({"error": "not found"}, 404)
            if path.startswith("/archive/"):
                from .highlights import ARCHIVE
                target = (ARCHIVE / path[len("/archive/"):]).resolve()
                if ARCHIVE.resolve() not in target.parents:
                    return self._json({"error": "forbidden"}, 403)
                return self._file(target)
            m = re.match(r"^/files/([\w.-]+)/(.+)$", path)
            if m:
                base = (mgr.ws / m.group(1)).resolve()
                target = (base / m.group(2)).resolve()
                if base not in target.parents:
                    return self._json({"error": "forbidden"}, 403)
                return self._file(target)
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            data = self._body()
            try:
                if path == "/api/jobs":
                    ref = (data.get("source") or "").strip()
                    if not ref:
                        raise TwitchCutError("Укажите ссылку на VOD или путь к файлу")
                    jid = mgr.submit(ref, data.get("options") or {})
                    return self._json({"id": jid})
                m = re.match(r"^/api/jobs/([\w.-]+)/rerun$", path)
                if m:
                    st = read_json(mgr.ws / m.group(1) / "job.json") or {}
                    opts = {**(st.get("options") or {}), **(data.get("options") or {})}
                    stage = data.get("from") if data.get("from") in STAGES else None
                    jid = mgr.submit(st.get("ref") or "", opts, stage)
                    return self._json({"id": jid})
                if path == "/api/streamers":
                    return self._json(streamers.upsert(data))
                if path == "/api/feedback":
                    return self._json(mgr.save_feedback(data))
                if path == "/api/cleanup":
                    from . import cleanup
                    days = float(data.get("days") if data.get("days") is not None else
                                 load_config(mgr.config_path).get("cleanup", {}).get("keep_days", 2))
                    return self._json(cleanup.cleanup(mgr.ws, days, set(mgr.live_jobs)))
                if path == "/api/feedback/refresh":
                    from . import feedback, tiktok
                    ids = {f["id"] for f in feedback.load() if f.get("status") == "posted" and tiktok.is_tiktok(f.get("url") or "")}
                    if data.get("id"):
                        ids = {data["id"]} & ids
                    threading.Thread(target=lambda: tiktok.refresh(ids), daemon=True).start()
                    return self._json({"ok": True, "count": len(ids)})
                if path == "/api/twitch/auth":
                    from .twitch_api import save_auth
                    return self._json(save_auth(data.get("text") or ""))
                if path == "/api/twitch/auth/clear":
                    from .twitch_api import clear_auth
                    clear_auth()
                    return self._json({"ok": True})
                m = re.match(r"^/api/streamers/([\w]+)/learn$", path)
                if m:
                    from . import feedback
                    cfg = load_config(mgr.config_path)
                    d = ROOT_DIR / "data"
                    res = feedback.learn_profile(m.group(1).lower(), cfg, d, force=True)
                    if not res:
                        raise TwitchCutError("Мало отметок: нужно хотя бы 4 «Выложил» / «Не подходит» у этого стримера")
                    return self._json(res)
                m = re.match(r"^/api/streamers/([\w]+)/delete$", path)
                if m:
                    streamers.delete(m.group(1))
                    return self._json({"ok": True})
                m = re.match(r"^/api/jobs/([\w.-]+)/clips/(\d+)/(render|shorten|restore)$", path)
                if m:
                    sp = float(data.get("speed") or 1.0)
                    lay = data.get("layout") if data.get("layout") in ("auto", "split", "cam", "screen", "crop", "blur") else None
                    tgt = None
                    if m.group(3) == "shorten":
                        tgt = float(data.get("seconds") or 0)
                        if not 5 <= tgt <= 600:
                            raise TwitchCutError("Укажите длину от 5 до 600 секунд")
                    mgr.rerender_clip(m.group(1), int(m.group(2)), sp, bool(data.get("trim", True)), lay,
                                      action=m.group(3), target=tgt)
                    return self._json({"ok": True})
                if path == "/api/highlights/build":
                    mgr.best_build(data)
                    return self._json({"ok": True})
                if path == "/api/highlights/extract":
                    mgr.best_extract(data.get("job") or "")
                    return self._json({"ok": True})
                if path == "/api/highlights/delete":
                    from . import highlights
                    highlights.delete(data.get("id") or "")
                    return self._json({"ok": True})
                if path == "/api/highlights/compilation/delete":
                    from . import highlights
                    comps = read_json(highlights.ARCHIVE / "compilations.json") or []
                    for c in [c for c in comps if c["file"] == data.get("file")]:
                        (ROOT_DIR / c["file"]).unlink(missing_ok=True)
                        (ROOT_DIR / c["thumb"]).unlink(missing_ok=True)
                    write_json(highlights.ARCHIVE / "compilations.json", [c for c in comps if c["file"] != data.get("file")])
                    return self._json({"ok": True})
                if path == "/api/highlights/compilation/tiktok":
                    from . import highlights, tiktok_upload
                    comps = read_json(highlights.ARCHIVE / "compilations.json") or []
                    c = next((x for x in comps if x["file"] == data.get("file")), None)
                    if not c:
                        raise TwitchCutError("Подборка не найдена")
                    acc = data.get("open_id") or tiktok_upload.account_for(c.get("streamer") or None)
                    res = tiktok_upload.upload_draft(ROOT_DIR / c["file"], acc)
                    tiktok_upload.register(res, acc, "best", "", c["file"], c.get("duration"), c.get("streamer"))
                    c["tiktok_draft"] = res
                    write_json(highlights.ARCHIVE / "compilations.json", comps)
                    return self._json(res)
                if path == "/api/tiktok/app":
                    from . import tiktok_upload
                    tiktok_upload.save_app(data.get("client_key"), data.get("client_secret"),
                                           data.get("video_list") if "video_list" in data else None)
                    return self._json(tiktok_upload.status())
                if path == "/api/tiktok/check":
                    from . import tiktok_upload
                    return self._json(tiktok_upload.track_once(mgr.on_tiktok_published, force=True))
                if path == "/api/tiktok/disconnect":
                    from . import tiktok_upload
                    tiktok_upload.disconnect(data.get("open_id") or None)
                    return self._json(tiktok_upload.status())
                if path == "/api/tiktok/assign":
                    from . import tiktok_upload
                    login = streamers.normalize_login(data.get("streamer") or "")
                    streamers.set_fields(login, tiktok_account=data.get("open_id") or "")
                    return self._json(tiktok_upload.status())
                m = re.match(r"^/api/jobs/([\w.-]+)/clips/(\d+)/tiktok$", path)
                if m:
                    return self._json(mgr.send_to_tiktok(m.group(1), n=int(m.group(2)), account=data.get("open_id")))
                m = re.match(r"^/api/jobs/([\w.-]+)/compilation$", path)
                if m:
                    n = int(data.get("n") or 5)
                    mgr.make_compilation(m.group(1), max(2, min(10, n)))
                    return self._json({"ok": True})
                m = re.match(r"^/api/jobs/([\w.-]+)/compilation/tiktok$", path)
                if m:
                    return self._json(mgr.send_to_tiktok(m.group(1), file=data.get("file"), account=data.get("open_id")))
                m = re.match(r"^/api/jobs/([\w.-]+)/stop$", path)
                if m:
                    mgr.stop_live(m.group(1))
                    return self._json({"ok": True})
                m = re.match(r"^/api/jobs/([\w.-]+)/delete$", path)
                if m:
                    mgr.delete_job(m.group(1))
                    return self._json({"ok": True})
                if path == "/api/status/refresh":
                    from .llm import claude_status
                    claude_status(force=True)
                    return self._json(mgr.status())
                m = re.match(r"^/api/jobs/([\w.-]+)/llm$", path)
                if m:
                    mgr.manual_answer(m.group(1), data.get("text") or "")
                    return self._json({"ok": True})
            except TwitchCutError as e:
                return self._json({"error": str(e)}, 400)
            return self._json({"error": "not found"}, 404)

    return H


def serve(host: str = "127.0.0.1", port: int = 8765, config_path: Path | None = None) -> None:
    mgr = Manager(config_path)
    mgr.port = port
    from . import tiktok
    tiktok.start_background()  # просмотры выложенных роликов обновляются сами
    from . import tiktok_upload
    tiktok_upload.start_tracker(mgr.on_tiktok_published)  # ссылки на опубликованные черновики ставятся сами
    class _Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            import sys as _sys
            if isinstance(_sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                return  # браузер оборвал соединение — не засоряем консоль
            super().handle_error(request, client_address)

    httpd = _Server((host, port), make_handler(mgr))
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    log.info("TwitchCut запущен: %s  (Ctrl+C — остановить)", url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
