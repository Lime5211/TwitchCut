"""Прямой эфир: пока стример в эфире, TwitchCut каждые ~10 минут берёт новый кусок записи
(Twitch пишет VOD параллельно с эфиром), анализирует его тем же конвейером и сразу монтирует
сильные моменты. Клипы появляются примерно через 10–20 минут после того, как что-то произошло.

Нагрузка: между проверками — почти ноль. Раз в ~10–15 минут: скачивается ~7 МБ звука (сегменты записи
напрямую по плейлисту, обычно меньше минуты), 1–3 минуты распознаётся речь, 1–2 запроса к Claude,
по ~1 минуте на монтаж клипа.
"""
from __future__ import annotations

import copy
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import chat as chatmod
from .llm import rank_candidates
from .pipeline import Job, heavy
from .scan import merge_candidates, scan_stream
from .signals import audio_loudness, find_candidates
from .transcribe import Transcriber
from .twitch_api import vod_status
from .util import Cancelled, TwitchCutError, fmt_time, log, read_json, write_json


class Stopped(Exception):
    pass


class LiveJob(Job):
    def __init__(self, vod_id: str, login: str, cfg: dict, on_update: Optional[Callable[[dict], None]] = None,
                 options: Optional[dict] = None):
        # отдельная папка live_<id>: обычная нарезка той же записи не затирает клипы эфира
        super().__init__(f"https://www.twitch.tv/videos/{vod_id}", cfg, on_update=on_update,
                         options={**(options or {}), "streamer": login}, job_id=f"live_{vod_id}")
        self.vod_id = vod_id
        self.login = login
        self.stop_event = threading.Event()
        self._tr: Optional[Transcriber] = None

    # ------------------------------------------------------------------ loop
    def run_live(self) -> None:
        lc = self.cfg["live"]
        try:
            vs = vod_status(self.vod_id)
        except TwitchCutError as e:
            self.update(status="error", error=str(e), message=str(e))
            return
        meta = read_json(self.dir / "meta.json") or {
            "kind": "twitch", "ref": self.source.ref, "vod_id": self.vod_id, "title": vs["title"],
            "channel": self.login, "channel_id": self.login, "duration": vs["length"], "live": True}
        write_json(self.dir / "meta.json", meta)
        self.update(status="live", live=True, title=meta["title"], channel=meta["channel"], error=None,
                    warning=None, stage="meta", message="Слежу за эфиром", live_min_score=lc["min_score"],
                    live_chunk_minutes=lc.get("max_chunk_minutes", 60))
        self._apply_streamer(meta, learn=True)
        if self.cfg["llm"]["backend"] == "manual":
            self.cfg["llm"]["backend"] = "none"  # в эфире вручную отвечать некому

        processed = float(self.state.get("processed_until") or 0.0)
        if not self.state.get("live_started_at"):
            self.update(live_started_at=processed)
        min_chunk = float(lc["min_chunk_minutes"]) * 60
        max_chunk = float(lc.get("max_chunk_minutes", 15)) * 60
        backfill = float(lc.get("backfill_minutes", 0) or 0) * 60  # 0 — весь эфир с самого начала
        if processed <= 0 and backfill > 0 and vs["length"] > backfill:
            # подключились посреди эфира: берём только последние backfill минут
            processed = vs["length"] - backfill
            self.update(processed_until=processed,
                        message=f"Подключился к эфиру на {fmt_time(vs['length'])}: начинаю с {fmt_time(processed)}")
        # куски, которые раньше не обработались из-за ошибки, — повторим
        retry = [list(x) for x in (self.state.get("live_retry") or [])]
        if not retry:
            hist = read_json(self.dir / "live_rankings.json") or []
            ok_a = {round(h["a"]) for h in hist if not h.get("error")}
            retry = [[h["a"], h["b"], 0] for h in hist if h.get("error") and round(h["a"]) not in ok_a]
            if retry:
                self.update(live_retry=retry)
        while not self.stop_event.is_set():
            try:
                vs = vod_status(self.vod_id)
            except TwitchCutError as e:
                self.update(message=f"Twitch временно недоступен: {e}")
                self.stop_event.wait(lc["poll_seconds"])
                continue
            recording = vs["status"] == "RECORDING"
            length = vs["length"]
            edge = length - 30 if recording else length
            self.update(duration=length, live_recording=recording)
            redo = next((r for r in retry if r[2] < 3), None)
            if redo is None and recording and edge - processed < min_chunk:
                wait = max(30, min_chunk - (edge - processed))
                self.update(stage="signals", status="live",
                            message=f"В эфире {fmt_time(length)} · обработано до {fmt_time(processed)} · "
                                    f"следующая проверка через ~{wait / 60:.0f} мин",
                            next_check=time.time() + min(wait, lc["poll_seconds"]))
                self.stop_event.wait(min(wait, lc["poll_seconds"]))
                continue
            if redo is None and edge - processed < 20:
                break  # эфир закончился и всё обработано
            if redo is not None:
                a, b = float(redo[0]), float(redo[1])
            else:
                a, b = processed, min(edge, processed + max_chunk)
            def waiting(owner: str):
                self.update(message=f"Новый кусок эфира готов, жду очереди: сейчас идёт {owner}")
            with heavy(f"эфир {self.login}", waiting):
                failed = False
                try:
                    self.cycle(meta, a, b, final=not recording or redo is not None)
                except Stopped:
                    break
                except Exception as e:  # не роняем наблюдение из-за одного куска
                    failed = True
                    log.exception("Эфир: кусок %s–%s не обработан", fmt_time(a), fmt_time(b))
                    self.update(warning=f"Кусок {fmt_time(a)}–{fmt_time(b)}: {e} — попробую ещё раз")
                    hist = read_json(self.dir / "live_rankings.json") or []
                    hist = [h for h in hist if round(h["a"]) != round(a) or not h.get("error")]
                    hist.append({"a": a, "b": b, "candidates": 0, "best": 0, "chosen": 0, "error": str(e)[:200]})
                    hist.sort(key=lambda h: h["a"])
                    write_json(self.dir / "live_rankings.json", hist[-200:])
                    self.update(live_cycles=[{k: h.get(k) for k in ("a", "b", "candidates", "best", "chosen", "error")}
                                             for h in hist[-40:]])
            if redo is not None:
                if failed:
                    redo[2] += 1
                else:
                    retry.remove(redo)
                self.update(live_retry=retry)
                if failed:
                    self.stop_event.wait(60)
                continue
            if failed:
                retry.append([a, b, 1])
                self.update(live_retry=retry)
            processed = b
            self.update(processed_until=processed)
            if not recording and edge - processed < 20 and not any(r[2] < 3 for r in retry):
                break
        if self.stop_event.is_set():
            clips = read_json(self.dir / "clips.json") or []
            self.update(status="done", live=False, live_stopped=True,
                        message=f"Наблюдение остановлено. Клипов: {len(clips)}")
        else:
            meta["duration"] = processed
            meta["live"] = False
            write_json(self.dir / "meta.json", meta)
            clips = read_json(self.dir / "clips.json") or []
            self.update(status="done", live=False, stage="render", stage_progress=1.0,
                        message=f"Эфир закончился. Клипов: {len(clips)}")

    def progress(self, stage: str, frac: float, msg: str = "") -> None:
        super().progress(stage, frac, msg)
        if self.stop_event.is_set():
            raise Stopped()  # «Остановить наблюдение» — прерываем на ближайшем шаге, а не после всего куска

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise Stopped()

    # ----------------------------------------------------------------- cycle
    def cycle(self, meta: dict, a: float, b: float, final: bool = False) -> None:
        lc = self.cfg["live"]
        n_cycle = int(self.state.get("live_cycle") or 0) + 1
        self.update(live_cycle=n_cycle, warning=None)
        tag = f"L{n_cycle}"
        log.info("Эфир %s: обрабатываю %s–%s", self.login, fmt_time(a), fmt_time(b))

        # 1. звук нового куска
        self.progress("audio", 0.1, f"Скачиваю звук {fmt_time(a)}–{fmt_time(b)}")
        self.source.refresh_variants()
        part = self.dir / "live_audio" / f"part_{int(a):06d}.m4a"
        part.parent.mkdir(exist_ok=True)
        for old in [*part.parent.glob(f"{part.stem}.part*"), *part.parent.glob(f"{part.stem}.dl*")]:
            try:
                old.unlink()  # хвосты прерванной загрузки
            except OSError:
                pass  # файл держит зависший процесс от прошлого запуска — не мешает, качаем в другой
        t_dl = time.time()

        def dl_progress(done: int, total: int) -> None:
            self.progress("audio", 0.1 + 0.15 * done / max(1, total),
                          f"Скачиваю звук {fmt_time(a)}–{fmt_time(b)}: {done} из {total} сегментов")

        self.source.fetch_audio_range(a, b, part, progress=dl_progress)
        log.info("Звук %s–%s скачан за %.0f с", fmt_time(a), fmt_time(b), time.time() - t_dl)

        self._check_stop()
        # 2. громкость — дописываем в общий ряд
        lp = self.dir / "loudness.npy"
        loud = np.load(lp) if lp.exists() else np.zeros(0, dtype=np.float32)
        part_loud = audio_loudness(part, b - a)
        need = int(a)
        if len(loud) < need:
            fill = float(np.median(loud)) if len(loud) else -40.0
            loud = np.concatenate([loud, np.full(need - len(loud), fill, dtype=np.float32)])
        tail = loud[need + len(part_loud):]  # при повторе старого куска не теряем то, что после него
        loud = np.concatenate([loud[:need], part_loud, tail]).astype(np.float32)
        np.save(lp, loud)

        # 3. чат
        self.progress("chat", 0.3, "Чат")
        chat = read_json(self.dir / "chat.json") or []
        try:
            new_chat = chatmod.fetch_chat_range(self.vod_id, max(0.0, a - 5), b)
            seen = {(round(m[0], 1), m[1], m[2]) for m in chat}
            chat += [m for m in new_chat if (round(m[0], 1), m[1], m[2]) not in seen]
            chat.sort(key=lambda m: m[0])
            write_json(self.dir / "chat.json", chat)
        except TwitchCutError as e:
            log.warning("Эфир: чат не получен: %s", e)
        self.update(chat_messages=len(chat))

        self._check_stop()
        # 4. речь
        self.progress("transcribe", 0.0, f"Распознаю речь {fmt_time(a)}–{fmt_time(b)}")
        if self._tr is None:
            self._tr = Transcriber(self.cfg)
        # кусками по 10 минут: видеокарта и процессор распознают разные куски одновременно
        # (на встроенной видеокарте одна она медленнее, чем вместе с процессором)
        t_tr = time.time()
        parts_dir = self.dir / "transcript_parts" / f"live_{int(a):06d}"

        def tr_progress(stage: str, frac: float, msg: str = "") -> None:
            # вызывается и из потока видеокарты — здесь не прерываем, остановку ловит само распознавание
            Job.progress(self, "transcribe", frac, msg.replace("всего стрима", f"{fmt_time(a)}–{fmt_time(b)}"))

        self._tr.cancel = self.stop_event
        try:
            res = self._tr.transcribe_full(part, b - a, parts_dir, tr_progress)
        except Cancelled:
            raise Stopped()
        words, segs = res["words"], res["segments"]
        log.info("Речь %s–%s распознана за %.0f с", fmt_time(a), fmt_time(b), time.time() - t_tr)
        for w in words:
            w["s"] = round(w["s"] + a, 2)
            w["e"] = round(w["e"] + a, 2)
        for sg in segs:
            sg["s"] = round(sg["s"] + a, 2)
            sg["e"] = round(sg["e"] + a, 2)
        tr = read_json(self.dir / "transcript.json") or {"words": [], "segments": [], "mode": "full"}
        tr["words"] = sorted([w for w in tr["words"] if w["s"] < a or w["s"] >= b] + words, key=lambda w: w["s"])
        tr["segments"] = sorted([s for s in tr["segments"] if s["s"] < a or s["s"] >= b] + segs, key=lambda s: s["s"])
        write_json(self.dir / "transcript.json", tr)

        self._check_stop()
        # 5. всплески чата/звука в новом куске
        meta_now = {**meta, "duration": b}
        sig_cands, timeline = find_candidates(b, loud, chat, [], self.cfg)
        done = self.state.get("live_done") or []
        sig_cands = [c for c in sig_cands if a - 45 <= c["peak"] <= b - 20
                     and not any(d[0] <= c["peak"] <= d[1] for d in done)]
        write_json(self.dir / "timeline.json", timeline)

        # 6. смысловой просмотр нового куска
        usage: dict = {}
        speech: list = []
        if self.cfg["scan"]["enabled"] and self.cfg["llm"]["backend"] in ("api", "claude-cli"):
            self.progress("scan", 0.0, "Claude читает новый кусок эфира")
            try:
                # Claude видит и 5 минут ДО нового куска: так понятно, о чём шла речь, и история,
                # начавшаяся в прошлом куске, не обрезается на стыке
                ctx = float(lc.get("context_minutes", 5)) * 60
                speech = scan_stream(meta_now, tr, chat, self.cfg, self.dir, usage, self.progress,
                                     start=max(0.0, a - ctx), end=b)
                speech = [s for s in speech if s["end"] > a]
                if not final:
                    # история упирается в конец куска — скорее всего ещё не закончилась:
                    # её целиком найдёт следующий кусок (он начинается с тех же 5 минут контекста)
                    speech = [s for s in speech if s["end"] < b - 25]
            except TwitchCutError as e:
                log.warning("Эфир: смысловой просмотр не удался: %s", e)
        cands = merge_candidates(sig_cands, speech, timeline, self.cfg) if speech else sig_cands
        for c in cands:
            c["id"] = f"{tag}{c['id']}"
        if not cands:
            hist = [h for h in (read_json(self.dir / "live_rankings.json") or []) if round(h["a"]) != round(a)]
            hist.append({"a": a, "b": b, "candidates": 0, "best": 0, "chosen": 0, "items": []})
            hist.sort(key=lambda h: h["a"])
            write_json(self.dir / "live_rankings.json", hist[-200:])
            self.update(message=f"Кусок {fmt_time(a)}–{fmt_time(b)}: ничего интересного",
                        live_cycles=[{k: h.get(k) for k in ("a", "b", "candidates", "best", "chosen", "error")}
                                     for h in hist[-40:]])
            return

        self._check_stop()
        # 7. оценка и монтаж сильных моментов
        ranking, u2 = rank_candidates(meta_now, cands, tr["words"], self.cfg, self.dir, self.progress)
        for k in ("input_tokens", "output_tokens"):
            usage[k] = usage.get(k, 0) + u2.get(k, 0)
        tot = self.state.get("llm_usage") or {"backend": self.cfg["llm"]["backend"]}
        for k in ("input_tokens", "output_tokens"):
            tot[k] = tot.get(k, 0) + usage.get(k, 0)
        tot["model"] = self.cfg["llm"].get("cli_model")
        if u2.get("warning"):
            self.update(warning=u2["warning"])

        clips = read_json(self.dir / "clips.json") or []
        saved_cfg = self.cfg
        self.cfg = copy.deepcopy(saved_cfg)
        self.cfg["clips"]["min_score"] = lc["min_score"]
        self.cfg["clips"]["fill_min_score"] = lc.get("fill_min_score", 50)
        self.cfg["clips"]["max_clips"] = lc["max_per_cycle"]
        try:
            chosen = self.select(cands, tr, ranking)
        finally:
            self.cfg = saved_cfg
        if self.cfg["llm"]["backend"] != "none":
            # Claude не смог оценить (не авторизован, лимит и т.п.) — такие «моменты» обычно непонятны
            # без контекста, в эфире их не монтируем
            blind = [c for c in chosen if (ranking.get(c.get("cand") or c.get("id")) or {}).get("heuristic")]
            if blind:
                log.warning("Эфир: %d момент(ов) без оценки Claude — не монтирую", len(blind))
                chosen = [c for c in chosen if c not in blind]
        chosen = [c for c in chosen if not any(c["start"] < x["end"] and x["start"] < c["end"] for x in clips)]
        strong = [c for c in chosen if not c.get("extra")]
        if strong:
            chosen = strong
        else:
            # сильных нет — берём лучший «запасной», если давно не было клипов: эфир не должен проходить впустую
            last_end = max([x.get("end", 0) for x in clips] + [float(self.state.get("live_started_at") or 0)])
            gap_min = float(lc.get("fill_every_minutes", 20)) * 60
            chosen = chosen[:1] if chosen and b - last_end >= gap_min else []
        for c in cands:
            done.append([c["start"], c["end"]])
        self.update(live_done=done[-400:], llm_usage=tot)
        if chosen:
            clips = self.render_list(chosen, tr["words"], len(clips) + 1, prev=clips)
            write_json(self.dir / "clips.json", clips)
            self.update(clips=clips)
        best = max((r.get("score", 0) for r in ranking.values()), default=0)
        log_p = self.dir / "live_rankings.json"
        hist = [h for h in (read_json(log_p) or []) if round(h["a"]) != round(a)]
        hist.append({"a": a, "b": b, "candidates": len(cands), "best": best, "chosen": len(chosen),
                     "items": [{"id": k, "score": v.get("score"), "title": v.get("title"), "category": v.get("category")}
                               for k, v in ranking.items()]})
        hist.sort(key=lambda h: h["a"])
        write_json(log_p, hist[-200:])
        self.update(live_cycles=[{k: h[k] for k in ("a", "b", "candidates", "best", "chosen")} | {
            "error": h.get("error")} for h in hist[-40:]])
        self.update(status="live", message=f"Кусок {fmt_time(a)}–{fmt_time(b)}: кандидатов {len(cands)}, "
                                           f"лучшая оценка {best}, смонтировано {len(chosen)} (порог {lc['min_score']})")
