"""Конвейер обработки VOD с кешированием каждой стадии.

Каждая стадия сохраняет результат в папку задачи, поэтому повторный запуск продолжает
с места остановки, а эксперименты с настройками (например, другой промпт или раскладка)
не требуют заново качать звук и распознавать речь.

Стадии: meta → chat → audio → signals → transcribe → scan → llm → render
"""
from __future__ import annotations

import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import chat as chatmod
from .llm import LONG_CATEGORIES, rank_candidates
from .profanity import ProfanityFilter
from .render import LAYOUT_VERSION, detect_layout, render_clip
from .signals import audio_loudness, find_candidates
from .sources import Source
from .transcribe import Transcriber, slice_words
from .util import (ROOT_DIR, TwitchCutError, fmt_time, log, read_json, safe_name, write_json)

STAGES = ["meta", "chat", "audio", "signals", "transcribe", "scan", "llm", "render"]
STAGE_TITLES = {
    "meta": "Информация о видео", "chat": "Чат", "audio": "Звук", "signals": "Поиск моментов",
    "transcribe": "Распознавание речи", "scan": "Поиск по смыслу", "llm": "Оценка Claude", "render": "Монтаж клипов",
}
# файлы, которые нужно удалить, чтобы перезапустить стадию
STAGE_OUTPUTS = {
    "meta": ["meta.json"], "chat": ["chat.json", "viewer_clips.json"], "audio": ["audio_src.*"],
    "signals": ["loudness.npy", "candidates.json", "timeline.json"], "transcribe": ["transcript.json", "transcript_parts"],
    "scan": ["speech_candidates.json", "all_candidates.json", "scan_parts", "scan_usage.json"],
    "llm": ["ranking.json", "llm_response.txt", "llm_prompt.txt", "rank_parts"], "render": ["clips.json", "clips", "clips.partial.json"],
}


# Тяжёлая обработка (распознавание, Claude, монтаж) идёт строго по одной, чтобы не перегружать ПК:
# обычные задачи и прямой эфир ждут друг друга.
HEAVY_LOCK = threading.Lock()
HEAVY_STATE: dict = {"owner": "", "since": 0.0}   # кто сейчас занимает обработку (для интерфейса)


class heavy:
    """Захват общей «очереди тяжёлой работы» с подписью, кто её занял."""

    def __init__(self, owner: str, on_wait=None):
        self.owner, self.on_wait = owner, on_wait

    def __enter__(self):
        if not HEAVY_LOCK.acquire(blocking=False):
            if self.on_wait:
                self.on_wait(HEAVY_STATE.get("owner") or "другая задача")
            HEAVY_LOCK.acquire()
        HEAVY_STATE.update(owner=self.owner, since=time.time())
        return self

    def __exit__(self, *exc):
        HEAVY_STATE.update(owner="", since=0.0)
        HEAVY_LOCK.release()
        return False


class Job:
    def __init__(self, ref: str, cfg: dict, chat_file: Optional[str] = None,
                 on_update: Optional[Callable[[dict], None]] = None,
                 wait_manual: Optional[Callable[[Path, Path], None]] = None,
                 options: Optional[dict] = None, job_id: Optional[str] = None):
        self.cfg = cfg
        self.options = options or {}
        self.source = Source(ref)
        self.id = job_id or self.source.job_id()
        self.dir = Path(cfg["workspace"]) / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chat_file = chat_file
        self.on_update = on_update
        self.wait_manual = wait_manual
        self.state = read_json(self.dir / "job.json") or {
            "id": self.id, "ref": self.source.ref, "created": time.time()}
        self.state.update({"ref": self.source.ref})
        self._last_save = 0.0
        self._lock = threading.RLock()

    # ------------------------------------------------------------- state
    def update(self, **kw) -> None:
        """Обновление состояния. Вызывается в том числе из потоков загрузчика — поэтому под замком
        и без исключений наружу (ошибка записи статуса не должна ломать скачивание)."""
        with self._lock:
            important = ("status" in kw) or (kw.get("stage") and kw.get("stage") != self.state.get("stage"))
            self.state.update(kw)
            self.state["updated"] = time.time()
            now = time.time()
            if important or now - self._last_save > 1.0:
                try:
                    write_json(self.dir / "job.json", self.state)
                    self._last_save = now
                except OSError as e:
                    log.debug("Не удалось сохранить статус: %s", e)
            snapshot = dict(self.state)
        if self.on_update:
            try:
                self.on_update(snapshot)
            except Exception:
                pass

    def progress(self, stage: str, frac: float, msg: str = "") -> None:
        self.update(stage=stage, stage_progress=round(frac, 3), message=msg)

    def invalidate_from(self, stage: str) -> None:
        idx = STAGES.index(stage)
        for st in STAGES[idx:]:
            for pat in STAGE_OUTPUTS[st]:
                for p in self.dir.glob(pat):
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)

    # --------------------------------------------------------------- run
    def run(self, force_from: Optional[str] = None) -> dict:
        if force_from:
            self.invalidate_from(force_from)
        def waiting(owner: str):
            self.update(status="queued", message=f"Жду своей очереди: сейчас идёт {owner}")
        with heavy(f"обработка «{self.state.get('title') or self.id}»", waiting):
            return self._run()

    def _run(self) -> dict:
        self.update(status="running", error=None, warning=None, stage="meta", message="Старт")
        t0 = time.time()
        try:
            meta = self._meta()
            chat = self._chat(meta)
            audio = self._audio()
            sig_cands, timeline = self._signals(meta, audio, chat)
            transcript = self._transcribe(meta, audio, sig_cands)
            cands = self._scan(meta, chat, transcript, sig_cands, timeline)
            ranking = self._llm(meta, cands, transcript)
            clips = self._render(meta, cands, transcript, ranking)
            self.update(status="done", stage="render", stage_progress=1.0,
                        message=f"Готово: {len(clips)} клипов за {fmt_time(time.time() - t0)}", clips=clips)
            return self.state
        except TwitchCutError as e:
            self.update(status="error", error=str(e), message=str(e))
            raise
        except Exception as e:
            self.update(status="error", error=f"{e}\n{traceback.format_exc()[-2000:]}", message=str(e))
            raise

    # ------------------------------------------------------------ stages
    def _meta(self) -> dict:
        p = self.dir / "meta.json"
        meta = read_json(p)
        if not meta:
            self.progress("meta", 0.2, "Получаю информацию о видео")
            meta = self.source.meta()
            write_json(p, meta)
        self.update(title=meta.get("title"), channel=meta.get("channel"), duration=meta.get("duration"))
        self._apply_streamer(meta, learn=True)
        log.info("Видео: %s | %s | %s", meta.get("channel"), meta.get("title"), fmt_time(meta.get("duration", 0)))
        return meta

    def _apply_streamer(self, meta: dict, learn: bool = False) -> None:
        """Подключает карточку стримера: выбранную вручную или найденную по каналу VOD.
        learn=True (в начале анализа) — сначала обновить «вкус канала» по новым отметкам автора."""
        from . import streamers
        login = self.options.get("streamer") or meta.get("channel_id") or ""
        st = streamers.get(login)
        if not st and meta.get("channel"):
            st = streamers.get(str(meta["channel"]).lower())
        if not st:
            if self.options.get("streamer_context"):
                self.cfg["llm"]["streamer_context"] = self.options["streamer_context"]
            return
        if learn and (self.cfg.get("learn") or {}).get("enabled", True) \
                and self.cfg["llm"]["backend"] in ("api", "claude-cli"):
            try:
                from . import feedback
                if feedback.learn_profile(st["login"], self.cfg, self.dir):
                    st = streamers.get(st["login"]) or st
            except Exception as e:
                log.warning("Не удалось обновить вкус канала: %s", e)
        streamers.apply_to_config(st, self.cfg, self.options.get("streamer_context") or "")
        # явный выбор в форме важнее настроек стримера
        if self.options.get("layout") and self.options["layout"] != "auto":
            self.cfg["render"]["layout"] = self.options["layout"]
        if self.options.get("max_clips"):
            self.cfg["clips"]["max_clips"] = int(self.options["max_clips"])
        self.update(streamer=st["login"], streamer_name=st.get("name"), streamer_avatar=st.get("avatar"))
        log.info("Карточка стримера: %s", st.get("name"))

    def _chat(self, meta: dict) -> list:
        p = self.dir / "chat.json"
        chat = read_json(p)
        # пустой чат от прошлой неудачной попытки — пробуем скачать ещё раз
        retry = chat == [] and self.source.kind == "twitch" and not self.chat_file
        if chat is None or retry:
            new_chat: list = []
            ok = True
            try:
                if self.chat_file:
                    new_chat = chatmod.load_chat_file(Path(self.chat_file))
                elif self.source.kind == "twitch":
                    self.progress("chat", 0.0, "Скачиваю чат")
                    new_chat = chatmod.fetch_chat_gql(self.source.vod_id, meta.get("duration", 0), self.progress)
                else:
                    log.info("Чат недоступен для этого источника — анализ только по звуку")
            except TwitchCutError as e:
                ok = False
                log.warning("Чат не получен: %s. Продолжаю без чата.", e)
                self.update(warning=f"Чат не получен ({e}). Моменты ищутся только по звуку — точность ниже. "
                                    "Можно скачать чат через TwitchDownloaderCLI и указать файл в «Дополнительно».")
            if ok:
                write_json(p, new_chat)
                if retry and new_chat:
                    # чат появился — пересчитываем всё, что считалось без него
                    self.invalidate_from("signals")
            chat = new_chat
            if not (self.dir / "viewer_clips.json").exists():
                tw = self.cfg["twitch"]
                vc = self.source.viewer_clips(tw.get("client_id", ""), tw.get("client_secret", ""))
                write_json(self.dir / "viewer_clips.json", vc)
        self.update(chat_messages=len(chat))
        return chat

    def _audio(self) -> Path:
        self.progress("audio", 0.0, "Готовлю звук")
        path = self.source.fetch_audio(self.dir, self.progress)
        self.progress("audio", 1.0, "Звук готов")
        return path

    def _signals(self, meta: dict, audio: Path, chat: list) -> tuple[list, dict]:
        cp = self.dir / "candidates.json"
        cands = read_json(cp)
        if cands is not None:
            return cands, read_json(self.dir / "timeline.json") or {}
        dur = meta.get("duration") or 0
        lp = self.dir / "loudness.npy"
        if lp.exists():
            loud = np.load(lp)
        else:
            self.progress("signals", 0.0, "Анализирую громкость")
            loud = audio_loudness(audio, dur, self.progress)
            np.save(lp, loud)
        if not dur:
            dur = float(len(loud))
            meta["duration"] = dur
            write_json(self.dir / "meta.json", meta)
        vc = read_json(self.dir / "viewer_clips.json") or []
        self.progress("signals", 0.8, "Ищу всплески реакций")
        cands, timeline = find_candidates(dur, loud, chat, vc, self.cfg, self.progress)
        write_json(cp, cands)
        write_json(self.dir / "timeline.json", timeline)
        self.update(candidates=len(cands))
        return cands, timeline

    def deep(self) -> bool:
        return self.cfg["analysis"]["mode"] == "deep"

    def _transcribe(self, meta: dict, audio: Path, cands: list) -> dict:
        p = self.dir / "transcript.json"
        tr = read_json(p)
        if tr is not None and self.deep() and tr.get("mode") != "full":
            log.info("Есть только частичная расшифровка — распознаю весь стрим для поиска по смыслу")
            tr = None
        if tr is None:
            self.progress("transcribe", 0.0, "Загружаю модель распознавания речи")
            t = Transcriber(self.cfg)
            if self.deep():
                tr = t.transcribe_full(audio, float(meta.get("duration") or 0), self.dir / "transcript_parts",
                                       self.progress)
            else:
                tr = t.transcribe_windows(audio, cands, self.progress)
            tr["mode"] = "full" if self.deep() else "windows"
            write_json(p, tr)
        return tr

    def _scan(self, meta: dict, chat: list, tr: dict, sig_cands: list, timeline: dict) -> list:
        """Смысловой просмотр всей расшифровки + объединение с кандидатами по чату/звуку."""
        from .scan import merge_candidates, scan_stream
        p = self.dir / "all_candidates.json"
        merged = read_json(p)
        if merged is not None:
            return merged
        speech: list = []
        usage: dict = {}
        can_scan = (self.deep() and self.cfg["scan"]["enabled"] and tr.get("mode") == "full"
                    and self.cfg["llm"]["backend"] in ("api", "claude-cli"))
        if can_scan:
            self.progress("scan", 0.0, "Claude читает весь стрим и ищет истории, новости, мнения…")
            try:
                speech = scan_stream(meta, tr, chat, self.cfg, self.dir, usage, self.progress)
            except TwitchCutError as e:
                log.warning("Смысловой просмотр не удался: %s", e)
                self.update(warning=f"Поиск по смыслу не удался ({e}). Моменты выбраны по чату и звуку.")
        write_json(self.dir / "speech_candidates.json", speech)
        write_json(self.dir / "scan_usage.json", usage)
        if usage.get("scan_failed"):
            self.update(warning="Не удалось просмотреть по смыслу куски " + ", ".join(usage["scan_failed"])
                        + " — остальной стрим проанализирован.")
        merged = merge_candidates(sig_cands, speech, timeline or {}, self.cfg) if speech else sig_cands
        write_json(p, merged)
        self.update(candidates=len(merged), speech_candidates=len(speech))
        return merged

    def _llm(self, meta: dict, cands: list, tr: dict) -> dict:
        p = self.dir / "ranking.json"
        data = read_json(p)
        if data is None:
            self.progress("llm", 0.0, "Анализ контекста")

            def wait(prompt_path: Path, resp_path: Path):
                self.update(status="awaiting_llm", message="Ждёт ответа Claude (ручной режим)",
                            llm_prompt_file=prompt_path.name)
                if not self.wait_manual:
                    raise TwitchCutError(f"Ручной режим: вставьте промпт из {prompt_path} в чат Claude и сохраните "
                                         f"ответ в {resp_path}, затем запустите задачу снова.")
                self.wait_manual(prompt_path, resp_path)
                self.update(status="running")

            ranking, usage = rank_candidates(meta, cands, tr["words"], self.cfg, self.dir, self.progress, wait)
            scan_u = read_json(self.dir / "scan_usage.json") or {}
            from .llm import USAGE_KEYS
            for k in USAGE_KEYS:
                usage[k] = usage.get(k, 0) + scan_u.get(k, 0)
            if scan_u.get("cost_usd") or usage.get("cost_usd") is not None:
                usage["cost_usd"] = round((usage.get("cost_usd") or 0) + (scan_u.get("cost_usd") or 0), 4)
            data = {"ranking": ranking, "usage": usage}
            write_json(p, data)
        usage = data.get("usage") or {}
        self.update(llm_usage=usage)
        if usage.get("warning"):
            self.update(warning=usage["warning"])
        return data["ranking"]

    def _priors(self) -> dict:
        from . import feedback
        try:
            return feedback.channel_priors(self.cfg["llm"].get("streamer_login") or self.state.get("streamer"))
        except Exception as e:
            log.warning("Статистика канала не прочитана: %s", e)
            return {}

    def candidate_item(self, c: dict, r: dict, words: list[dict], priors: dict) -> dict:
        """Кандидат + оценка Claude → готовое к монтажу описание клипа (границы, вырезки, итоговая оценка)."""
        from . import feedback
        cl = self.cfg["clips"]
        s, e = finalize_bounds(c, c["start"] + r["rel_start"], c["start"] + r["rel_end"], words, self.cfg,
                               long=r["category"] in LONG_CATEGORIES)
        # поправка по реальным просмотрам канала: тип ролика и примерная длина после вырезок
        est = (e - s) - sum(max(0.0, min(b, e) - max(a, s)) for a, b in
                            ((c["start"] + x, c["start"] + y) for x, y in (r.get("rel_cuts") or [])))
        bonus = (priors.get("cat") or {}).get(r["category"], 0.0) + \
            (priors.get("dur") or {}).get(feedback.dur_bucket(est * 0.93), 0.0)
        bonus = max(-6.0, min(6.0, bonus))
        score = r["score"] + bonus
        # главное — оценка Claude по смыслу; реакция чата/звука — небольшая добавка
        final = 0.88 * score + 12 * c.get("signal_norm", 0.3)
        strong = (r["keep"] or score >= 55) and score >= cl["min_score"]
        # вырезки «воды» от Claude (донат не по теме, повтор, поиск) — в абсолютном времени стрима
        cuts = []
        for a, b in r.get("rel_cuts") or []:
            a, b = max(s + 0.5, c["start"] + a), min(e - 0.5, c["start"] + b)
            if b - a >= 0.8:
                cuts.append([round(a, 2), round(b, 2)])
        if sum(b - a for a, b in cuts) > 0.6 * (e - s):
            cuts = []  # подозрительно много — не доверяем
        spans = [[round(max(s, c["start"] + a), 2), round(min(e, c["start"] + b), 2)]
                 for a, b in (r.get("rel_screen") or [])]
        spans = [x for x in spans if x[1] - x[0] >= 1.0]
        return {**r, "cand": c["id"], "source": c.get("source", "signals"), "topic": c.get("topic", ""),
                "start": s, "end": e, "final_score": round(final, 1), "extra": not strong,
                "channel_bonus": round(bonus, 1), "adj_score": round(score, 1),
                "content_cuts": cuts, "screen_spans": spans}

    def select(self, cands: list, tr: dict, ranking: dict) -> list[dict]:
        """Сильные моменты (оценка ≥ min_score); если их меньше, чем просили, — добираем лучшими
        из оставшихся (≥ fill_min_score), помечая их как «запасные»."""
        cl = self.cfg["clips"]
        words = tr["words"]
        floor = float(cl.get("fill_min_score", 40))
        priors = self._priors()
        picked = []
        for c in cands:
            r = ranking.get(c["id"])
            if not r:
                continue
            item = self.candidate_item(c, r, words, priors)
            if item["adj_score"] < floor:
                continue
            picked.append(item)
        picked.sort(key=lambda x: (x["extra"], -x["final_score"]))
        chosen: list[dict] = []
        for p in picked:
            if any(p["start"] < q["end"] - 3 and q["start"] < p["end"] - 3 for q in chosen):
                continue
            chosen.append(p)
            if len(chosen) >= cl["max_clips"]:
                break
        chosen.sort(key=lambda x: -x["final_score"])
        return chosen

    # ------------------------------------------------- все найденные моменты
    def moments(self) -> tuple[list[dict], dict]:
        """Все кандидаты, которые оценивал Claude, и его оценки (и для записи, и для эфира)."""
        cands = read_json(self.dir / "all_candidates.json") or read_json(self.dir / "live_candidates.json") or []
        ranking = (read_json(self.dir / "ranking.json") or {}).get("ranking") or {}
        ranking.update(read_json(self.dir / "live_ranking.json") or {})
        return cands, ranking

    def render_candidate(self, cid: str) -> dict:
        """Смонтировать любой найденный момент (даже не прошедший порог) — по уже готовой оценке Claude,
        без новых запросов к нему (кроме режиссёра для раскладки, как у обычного клипа)."""
        cands, ranking = self.moments()
        c = next((x for x in cands if x.get("id") == cid), None)
        r = ranking.get(cid)
        if not c or not r:
            raise TwitchCutError("Момент не найден (возможно, задачу анализировали до обновления)")
        meta = read_json(self.dir / "meta.json") or {}
        if meta:
            self._apply_streamer(meta)
        tr = read_json(self.dir / "transcript.json") or {"words": []}
        item = self.candidate_item(c, r, tr["words"], self._priors())
        item["manual"] = True
        clips = read_json(self.dir / "clips.json") or []
        if any(x.get("candidate") == cid for x in clips):
            raise TwitchCutError("Этот момент уже смонтирован")
        sc = read_json(self.dir / "stream_cam.json")
        if sc and not self.cfg["render"].get("cam_rect"):
            self.cfg["render"]["stream_cam"] = sc
        clips = self.render_list([item], tr["words"], len(clips) + 1, prev=clips)
        write_json(self.dir / "clips.json", clips)
        self._save_clips_state(clips)
        return next((x for x in clips if x.get("candidate") == cid), {})

    def _save_clips_state(self, clips: list) -> None:
        """Записать клипы в job.json, не затирая статус задачи: Job для ручного монтажа создаётся заранее
        и мог прочитать job.json задолго до того, как дошла очередь (за это время эфир успевал записать
        новый кусок или закончиться)."""
        with self._lock:
            fresh = read_json(self.dir / "job.json") or self.state
            fresh["clips"] = clips
            fresh["updated"] = time.time()
            self.state = fresh
            try:
                write_json(self.dir / "job.json", fresh)
            except OSError as e:
                log.debug("Не удалось сохранить статус: %s", e)

    def _render(self, meta: dict, cands: list, tr: dict, ranking: dict) -> list[dict]:
        p = self.dir / "clips.json"
        existing = read_json(p)
        if existing is not None:
            return existing
        chosen = self.select(cands, tr, ranking)
        if not chosen:
            log.warning("Ни один момент не прошёл порог оценки — снизьте clips.min_score")
        pp = p.with_suffix(".partial.json")
        if not pp.exists():
            self.update(render_failed={})
        clips = self.render_list(chosen, tr["words"], 1, partial_path=pp, prev=read_json(pp) or [])
        write_json(p, clips)
        p.with_suffix(".partial.json").unlink(missing_ok=True)
        return clips

    def reaction_peaks(self, start: float, end: float, max_n: int = 2) -> list[float]:
        """Моменты резкой реакции внутри клипа (по громкости): туда ставим «наезд» камеры.
        Возвращает время от начала клипа."""
        lp = self.dir / "loudness.npy"
        if not lp.exists():
            return []
        loud = np.load(lp)
        a, b = int(start), int(min(len(loud), end))
        if b - a < 6:
            return []
        seg = loud[a:b]
        base = float(np.median(seg))
        jump = np.diff(seg, prepend=seg[0])
        # резкий рост громкости и заметно громче фона клипа
        score = np.where((seg - base > 7) & (jump > 4), seg - base + jump, 0)
        peaks: list[float] = []
        for i in np.argsort(-score):
            if score[i] <= 0 or len(peaks) >= max_n:
                break
            t = a + i - start
            if 1.0 < t < (end - start) - 1.5 and all(abs(t - p) > 6 for p in peaks):
                peaks.append(float(t))
        return sorted(peaks)

    def _analysis_audio(self, t: float, need: float = 5.0) -> Optional[tuple[Path, float]]:
        """Аудиофайл, по которому распознавалась речь в момент t стрима, и позиция t внутри него."""
        from .sources import _audio_files
        files = _audio_files(self.dir)
        if files:  # обычный анализ записи: звук всего стрима
            return files[0], t
        parts = sorted((self.dir / "live_audio").glob("part_*.m4a"))
        for p in reversed(parts):  # эфир: куски звука part_<начало в секундах>.m4a
            if ".part" in p.name:
                continue
            try:
                a = float(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if a <= t:
                from .util import ffprobe_duration
                cache = self.__dict__.setdefault("_part_len", {})
                if p not in cache:
                    cache[p] = ffprobe_duration(p)
                return (p, t - a) if t - a <= cache[p] - need else None
        if self.source.kind == "file":
            return self.source.path, t
        return None

    def _synced_words(self, c: dict, words: list[dict], src: Path, offset: float) -> list[dict]:
        """Сверяет звук расшифровки со звуком видеофрагмента клипа и при необходимости поправляет субтитры."""
        from .sync import measure_shift
        dur = c["end"] - c["start"]
        sub_shift = float(self.cfg["render"].get("subtitle_shift") or 0.0)
        # звук расшифровки, который покрывает начало клипа (или чуть позже, если клип на стыке кусков)
        ref, seg_from = None, offset
        for d in (0.0, 30.0, 60.0):
            if c["end"] - (c["start"] + d) < 8:
                break
            ref = self._analysis_audio(c["start"] + d, need=min(45.0, c["end"] - c["start"] - d) + 1)
            if ref:
                seg_from = offset + d
                break
        if ref is None:
            return [{**w, "s": w["s"] - sub_shift, "e": w["e"] - sub_shift} for w in words] if sub_shift else words
        try:
            shift, conf = measure_shift(ref[0], ref[1], src, seg_from, c["end"] - c["start"] - (seg_from - offset))
        except Exception as e:
            log.warning("Проверка синхронности субтитров не удалась: %s", e)
            return words
        if conf >= 0.35:
            # звук совпал: слово, сказанное в расшифровке в момент t, в видео звучит в t − shift
            if abs(shift) >= 0.08:
                log.info("Субтитры клипа %s: сдвиг %.2f с (уверенность %.2f) — поправляю",
                         c.get("cand"), shift, conf)
            total = shift + sub_shift
            if abs(total) < 0.01:
                return words
            return [{**w, "s": w["s"] - total, "e": w["e"] - total} for w in words]
        if conf >= 0.15:
            log.info("Субтитры клипа %s: совпадение звука неуверенное (%.2f), оставляю как есть", c.get("cand"), conf)
            return [{**w, "s": w["s"] - sub_shift, "e": w["e"] - sub_shift} for w in words] if sub_shift else words
        # звук расшифровки не от этого места — распознаём речь прямо по видеофрагменту клипа
        log.warning("Субтитры клипа %s: расшифровка не совпадает со звуком видео (%.2f) — распознаю речь клипа заново",
                    c.get("cand"), conf)
        try:
            tr = getattr(self, "_tr", None) or getattr(self, "_sub_tr", None)
            if tr is None:
                tr = Transcriber(self.cfg)
                self._sub_tr = tr
            ws, _ = tr.transcribe_range(src, offset, offset + dur)
        except Exception as e:
            log.warning("Не удалось распознать речь клипа заново: %s", e)
            return words
        base = c["start"] - offset
        fresh = [{**w, "s": round(w["s"] + base - sub_shift, 2), "e": round(w["e"] + base - sub_shift, 2)} for w in ws]
        return [w for w in words if w["e"] <= c["start"] or w["s"] >= c["end"]] + fresh

    def _retimed_words(self, c: dict, words: list[dict], src: Path, offset: float) -> list[dict]:
        """Точный тайминг слов для субтитров. Основное распознавание (whisper.cpp на видеокарте) ставит
        время слов внутри фразы приблизительно — субтитры то торопятся, то догоняют. Здесь звук самого
        клипа ещё раз распознаётся на процессоре (faster-whisper находит, где в звуке каждое слово),
        и точное время переносится на слова субтитров; текст остаётся прежним."""
        from .sync import retime_words
        dur = c["end"] - c["start"]
        t0 = time.time()
        try:
            tr = getattr(self, "_tr", None) or getattr(self, "_sub_tr", None)
            if tr is None:
                tr = Transcriber(self.cfg)
                self._sub_tr = tr
            fresh, _ = tr._cpu_range(src, offset, offset + dur)
        except Exception as e:
            log.warning("Уточнение тайминга субтитров недоступно: %s", e)
            self.cfg["render"]["subtitle_retime"] = False  # не пытаемся на каждом клипе
            return words
        base = c["start"] - offset
        fresh = [{**w, "s": w["s"] + base, "e": w["e"] + base} for w in fresh]
        new, rep = retime_words(words, fresh, c["start"], c["end"])
        if rep["applied"]:
            log.info("Субтитры клипа %s: тайминг уточнён (совпало %d из %d слов, типичный сдвиг %+.2f с, "
                     "до %.2f с) за %.0f с", c.get("cand"), rep["matched"], rep["n"], rep["shift"],
                     rep.get("spread", 0), time.time() - t0)
            return new
        log.info("Субтитры клипа %s: тайминг не уточнён (совпало %d из %d слов)", c.get("cand"), rep["matched"], rep["n"])
        return words

    def _render_one(self, c: dict, words: list[dict], src: Path, offset: float, out: Path,
                    prof: ProfanityFilter, speed: float, trim: bool, director_mode: str = "new") -> tuple[dict, list]:
        """Монтаж одного клипа. director_mode: new — режиссёр как обычно; cache — сначала готовый план из
        прошлого монтажа (перемонтаж по кнопке), cache_only — вообще без запросов к Claude."""
        reuse_only = director_mode == "cache_only"
        from .render import find_pauses
        dur = c["end"] - c["start"]
        words = self._synced_words(c, slice_words(words, c["start"] - 15, c["end"] + 15), src, offset)
        if self.cfg["render"].get("subtitle_retime", True):
            words = self._retimed_words(c, words, src, offset)
        words_rel = []
        for w in slice_words(words, c["start"], c["end"]):
            mid = (w["s"] + w["e"]) / 2
            if c["start"] <= mid <= c["end"]:
                words_rel.append({"w": w["w"], "s": max(0.0, w["s"] - c["start"]), "e": min(dur, w["e"] - c["start"])})
        keep = None
        if trim:
            try:
                keep = find_pauses(src, offset, dur, words_rel, c.get("category") or "other", self.cfg)
            except Exception as e:
                log.warning("Поиск пауз не удался: %s", e)
        cuts = [(a - c["start"], b - c["start"]) for a, b in (c.get("content_cuts") or [])]
        if cuts:
            keep = subtract_intervals(keep or [(0.0, dur)], cuts)
            # слова внутри вырезанного не нужны в субтитрах
            words_rel = [w for w in words_rel if not any(a <= (w["s"] + w["e"]) / 2 <= b for a, b in cuts)]
        hook = c.get("hook", "") if self.cfg["render"].get("hook_title") else ""
        hint = {"visual": c.get("visual"),
                "spans": [(a - c["start"], b - c["start"]) for a, b in (c.get("screen_spans") or [])]}
        li, dplan = None, None
        from . import director
        if self.cfg["render"].get("layout") == "auto" and director.enabled(self.cfg):
            try:
                li = detect_layout(src, offset, dur, self.cfg, words_rel, hint)
                rc = self.cfg.get("director") or {}
                # стример просто рассказывает (так решил Claude по смыслу), а экран весь клип стоит —
                # и без режиссёра ясно, что нужна только вебка; не тратим на это запрос с картинками
                static_talk = (rc.get("skip_static_talk", True) and c.get("visual") == "talk"
                               and float(li.get("screen_active", 1.0)) <= 0.1)
                cache = self.dir / "segments" / f"{c.get('cand') or c.get('candidate')}.director.json"
                planned = director.cached_plan(cache, dur) if director_mode != "new" else None
                if li.get("cam") and planned:
                    dplan = planned  # перемонтаж: режиссёр уже решал по этому клипу — токены не тратим
                elif li.get("cam") and not reuse_only and not static_talk:  # есть вебка — решаем, глядя на кадры
                    usage = self.state.get("director_usage") or {}
                    about = f"О чём: «{c.get('title') or ''}» ({c.get('category') or ''}). {c.get('reason') or ''}"
                    vis = c.get("visual")
                    if vis == "talk":
                        about += ("\nПодсказка оценщика: стример РАССКАЗЫВАЕТ/рассуждает о своём — экран для смысла не "
                                  "нужен, по умолчанию \"cam\" (или \"camwide\", если жестикулирует/показывает в камеру).")
                    elif vis == "screen":
                        about += "\nПодсказка оценщика: смысл в том, что на экране — нужен \"split\" с focus на важном."
                    elif vis == "mixed" and hint.get("spans"):
                        about += ("\nПодсказка оценщика: экран нужен только в эти секунды от начала клипа: "
                                  + ", ".join(f"{max(0.0, a):.0f}–{b:.0f}" for a, b in hint["spans"])
                                  + "; в остальное время — вебка.")
                    from . import feedback
                    notes = feedback.edit_notes(self.cfg["llm"].get("streamer_login"))
                    dplan = director.plan_clip(src, offset, dur, words_rel, keep or [(0.0, dur)], li, self.cfg,
                                               self.dir, usage, about=about, notes=notes, cache=cache)
                    self.update(director_usage=usage)
                elif li.get("cam") and static_talk and not reuse_only:
                    log.info("Клип %s: рассказ при неподвижном экране — только вебка, без режиссёра",
                             c.get("cand") or c.get("candidate"))
                if dplan:
                    if dplan["cuts"]:
                        keep = subtract_intervals(keep or [(0.0, dur)], [tuple(x) for x in dplan["cuts"]])
                        words_rel = [w for w in words_rel
                                     if not any(a <= (w["s"] + w["e"]) / 2 <= b for a, b in dplan["cuts"])]
                        cuts = cuts + [tuple(x) for x in dplan["cuts"]]
                    li = director.apply_plan(li, dplan)
            except Exception as e:  # режиссёр — улучшение, а не обязательный шаг
                log.warning("ИИ-режиссёр не сработал для клипа %s: %s — раскладка по кадрам", c.get("cand"), e)
                dplan = None
        info = render_clip(src, offset, dur, words_rel, hook, out, self.cfg, prof, layout_info=li,
                           zooms=self.reaction_peaks(c["start"], c["end"]), keep=keep, speed=speed, layout_hint=hint)
        info["director"] = bool(dplan)
        info["layout_v"] = LAYOUT_VERSION
        from .render import people_shown
        info["people"] = people_shown(li) if li else 1
        if dplan:
            info["director_shots"] = [{k: s_[k] for k in ("t0", "t1", "layout", "focus", "why")} for s_ in dplan["shots"]]
        info["content_cut_sec"] = round(sum(b - a for a, b in cuts), 1)
        info["pauses_removed"] = round(max(0.0, info.get("pauses_removed", 0) - info["content_cut_sec"]), 1)
        return info, words_rel

    def shorten_clip(self, n: int, target: float, speed: float, trim: bool, layout: str | None = None) -> dict:
        """«Ужать до N с»: Claude оставляет только самое сильное (новые границы + вырезки), затем клип
        перемонтируется. Прежние границы сохраняются, чтобы можно было вернуть."""
        from .llm import shorten_plan
        p = self.dir / "clips.json"
        clips = read_json(p) or []
        c = next((x for x in clips if x.get("n") == n), None)
        if not c:
            raise TwitchCutError("Клип не найден")
        meta = read_json(self.dir / "meta.json") or {}
        if meta:
            self._apply_streamer(meta)
        tr = read_json(self.dir / "transcript.json") or {"words": []}
        plan = shorten_plan(c, tr["words"], target, self.cfg, self.dir)
        st = float(c["start"])
        new_cuts = [[round(st + a, 2), round(st + b, 2)] for a, b in plan["cuts"]]
        for x in clips:
            if x.get("n") == n:
                x.setdefault("original", {"start": x["start"], "end": x["end"],
                                          "content_cuts": x.get("content_cuts") or []})
                x.update(start=round(st + plan["start"], 2), end=round(st + plan["end"], 2), content_cuts=new_cuts,
                         shorten_target=int(target), shorten_note=plan["note"])
                x["duration"] = round(x["end"] - x["start"], 1)
                x["stream_time"] = fmt_time(x["start"])
        write_json(p, clips)
        log.info("Клип %s ужат: %.0f–%.0f с от начала, вырезок %d (цель %d с)", n, plan["start"], plan["end"],
                 len(new_cuts), target)
        return self.rerender_clip(n, speed, trim, layout)

    def restore_clip(self, n: int, speed: float, trim: bool, layout: str | None = None) -> dict:
        """Вернуть границы клипа, какие были до «ужать»."""
        p = self.dir / "clips.json"
        clips = read_json(p) or []
        for x in clips:
            if x.get("n") == n and x.get("original"):
                x.update(x.pop("original"))
                x.pop("shorten_target", None)
                x.pop("shorten_note", None)
                x["duration"] = round(x["end"] - x["start"], 1)
                x["stream_time"] = fmt_time(x["start"])
        write_json(p, clips)
        return self.rerender_clip(n, speed, trim, layout)

    def rerender_clip(self, n: int, speed: float, trim: bool, layout: str | None = None,
                      director_mode: str = "cache") -> dict:
        """Перемонтировать один готовый клип: другое ускорение и/или вырезание пауз. Решение режиссёра
        берётся из прошлого монтажа (если клип не ужимали) — Claude заново не спрашиваем."""
        p = self.dir / "clips.json"
        clips = read_json(p) or []
        c = next((x for x in clips if x.get("n") == n), None)
        if not c:
            raise TwitchCutError("Клип не найден")
        tr = read_json(self.dir / "transcript.json") or {"words": []}
        meta = read_json(self.dir / "meta.json") or {}
        if meta:
            self._apply_streamer(meta)  # раскладка/вебка из карточки стримера
        prof = ProfanityFilter(ROOT_DIR / self.cfg["censor"]["extra_words_file"])
        src, offset = self.source.fetch_segment(c["start"], c["end"], self.dir / "segments" / f"{c['candidate']}.mkv",
                                                int(self.cfg["render"]["quality"]))
        out = self.dir / c["file"]
        sc = read_json(self.dir / "stream_cam.json")
        if sc and not self.cfg["render"].get("cam_rect"):
            self.cfg["render"]["stream_cam"] = sc
        layout = layout or c.get("layout_choice") or "auto"
        saved = self.cfg["render"]["layout"]
        if layout != "auto":
            self.cfg["render"]["layout"] = layout
        try:
            info, _ = self._render_one({**c, "cand": c["candidate"]}, tr["words"], src, offset, out, prof,
                                       max(0.5, min(2.0, speed)), trim, director_mode=director_mode)
        finally:
            self.cfg["render"]["layout"] = saved
        info["layout_choice"] = layout
        clips = read_json(p) or clips
        for x in clips:
            if x.get("n") == n:
                x.update(info)
                x.update(trim_pauses=trim, v=int(time.time()))
        write_json(p, clips)
        return next(x for x in clips if x.get("n") == n)

    def relayout_outdated(self, skip: set | None = None, on_clip=None) -> int:
        """После обновления раскладки: перемонтировать готовые клипы этой задачи по-новому — без запросов
        к Claude (оценка, границы, вырезки и решения режиссёра — из прошлого монтажа). Клип монтируется
        заново, только если новая раскладка что-то меняет (например, в кадре несколько человек);
        остальным просто ставится отметка новой версии. skip — номера клипов, которые не трогать
        (уже выложенные). Возвращает, сколько клипов перемонтировано."""
        p = self.dir / "clips.json"
        clips = read_json(p) or []
        todo = [c for c in clips if int(c.get("layout_v") or 1) < LAYOUT_VERSION and c.get("n") not in (skip or set())
                and c.get("candidate")]
        if not todo:
            return 0
        meta = read_json(self.dir / "meta.json") or {}
        if meta:
            self._apply_streamer(meta)
        sc = read_json(self.dir / "stream_cam.json")
        if sc and not self.cfg["render"].get("cam_rect"):
            self.cfg["render"]["stream_cam"] = sc
        done = 0
        for c in todo:
            seg_meta = read_json(self.dir / "segments" / f"{c['candidate']}.json") or {}
            src = Path(seg_meta.get("file") or "")
            stamp = {"layout_v": LAYOUT_VERSION}
            if not seg_meta or not src.exists():
                stamp["relayout_skipped"] = "видео фрагмента уже удалено автоочисткой"
            else:
                try:
                    offset = float(c["start"]) - float(seg_meta["t0"])
                    li = detect_layout(src, offset, float(c["end"]) - float(c["start"]), self.cfg)
                    from .render import people_shown
                    # перемонтируем, если сейчас в кадре несколько человек или прошлый монтаж ошибочно
                    # показывал нескольких (например, лицо из видео на экране)
                    if people_shown(li) >= 2 or int(c.get("people") or 1) >= 2:
                        if on_clip:
                            on_clip(c)
                        log.info("Перемонтаж клипа %s (%s) по новой раскладке", c.get("n"), self.dir.name)
                        self.rerender_clip(c["n"], float(c.get("speed") or 1.0), bool(c.get("trim_pauses", True)),
                                           c.get("layout_choice"), director_mode="cache_only")
                        stamp["relayout"] = int(time.time())
                        done += 1
                except Exception as e:
                    log.warning("Перемонтаж клипа %s (%s) не удался: %s", c.get("n"), self.dir.name, e)
                    stamp["relayout_skipped"] = str(e)[:200]
            clips = read_json(p) or clips
            for x in clips:
                if x.get("n") == c["n"]:
                    x.update(stamp)
            write_json(p, clips)
        return done

    def render_list(self, chosen: list[dict], words: list[dict], n0: int, partial_path: Optional[Path] = None,
                    prev: Optional[list] = None) -> list[dict]:
        """Монтирует выбранные моменты. Используется и для VOD, и для прямого эфира."""
        out_dir = self.dir / "clips"
        seg_dir = self.dir / "segments"
        out_dir.mkdir(exist_ok=True)
        seg_dir.mkdir(exist_ok=True)
        prof = ProfanityFilter(ROOT_DIR / self.cfg["censor"]["extra_words_file"])
        clips = list(prev or [])
        done_c = {x.get("candidate") for x in clips}
        todo = [c for c in chosen if c["cand"] not in done_c]  # после перезапуска — только несмонтированные
        # 1. скачиваем куски видео всех клипов
        srcs: dict = {}
        failed: dict = dict(self.state.get("render_failed") or {})
        for k, c in enumerate(todo):
            self.progress("render", 0.3 * k / max(1, len(todo)), f"Скачиваю видео клипа {k + 1}/{len(todo)}")
            try:
                srcs[c["cand"]] = self.source.fetch_segment(c["start"], c["end"], seg_dir / f"{c['cand']}.mkv",
                                                            int(self.cfg["render"]["quality"]))
            except Exception as e:
                log.warning("Не удалось скачать видео момента %s: %s", c["cand"], e)
                failed[c["cand"]] = f"скачивание: {e}"[:300]
            if k == 0 and self.source.kind == "twitch":
                from .sources import VARIANTS_DIAG
                self.update(quality=self.source.last_quality, quality_diag=dict(VARIANTS_DIAG))
        # 2. где вебка — по всем клипам сразу (а не по одному: так не спутать с лицом из ролика на экране)
        self._prepare_stream_cam(todo, srcs)
        # 3. монтаж
        i = max([x.get("n", 0) for x in clips] + [n0 - 1])
        for k, c in enumerate(todo):
            if c["cand"] not in srcs:
                continue
            src, offset = srcs[c["cand"]]
            self.progress("render", 0.3 + 0.7 * k / max(1, len(todo)),
                          f"Монтаж клипа {k + 1}/{len(todo)}: {c.get('title') or c['cand']}")
            dur = c["end"] - c["start"]
            name = f"{i + 1:02d}_{safe_name(c.get('title') or c['cand'], 50).replace(' ', '_')}"
            out = out_dir / f"{name}.mp4"
            speed = float(self.cfg["render"].get("speed") or 1.0)
            trim = bool(self.cfg["render"].get("trim_pauses", True))
            try:
                info, words_rel = self._render_one(c, words, src, offset, out, prof, speed, trim)
            except Exception as e:  # один неудачный клип не должен останавливать остальные
                log.error("Монтаж клипа %s не удался: %s", c["cand"], e)
                failed[c["cand"]] = f"монтаж: {e}"[:300]
                self.update(render_failed=failed)
                continue
            i += 1
            from . import streamers as _streamers
            login = self.cfg["llm"].get("streamer_login") or self.state.get("streamer") or ""
            tags = _streamers.tiktok_caption(login, c.get("hashtags", [])) if login else \
                " ".join("#" + h.replace(" ", "") for h in c.get("hashtags", []))
            titles = c.get("titles") or ([c["title"]] if c.get("title") else [])
            (out_dir / f"{name}.txt").write_text(
                "Варианты названия:\n" + "\n".join(f"- {t}" for t in titles) + f"\n\nОписание для TikTok:\n{tags}\n", encoding="utf-8")
            clips.append({
                "n": i, "file": f"clips/{out.name}", "thumb": f"clips/{out.with_suffix('.jpg').name}",
                "title": c.get("title"), "titles": titles, "description": c.get("description"),
                "source": c.get("source"), "topic": c.get("topic"),
                "hashtags": c.get("hashtags", []), "score": c["score"], "final_score": c["final_score"],
                "category": c["category"], "reason": c.get("reason"), "start": round(c["start"], 2),
                "end": round(c["end"], 2), "duration": round(dur, 1), "stream_time": fmt_time(c["start"]),
                "candidate": c["cand"], "quality": self.source.last_quality or info.get("src_quality", ""), **info,
                "text": " ".join(w["w"] for w in words_rel)[:600],
                "trim_pauses": trim, "content_cuts": c.get("content_cuts") or [], "created": time.time(), "v": int(time.time()), "extra": c.get("extra", False),
                "visual": c.get("visual"), "screen_spans": c.get("screen_spans") or [],
            })
            if partial_path:
                write_json(partial_path, clips)
        if failed:
            self.update(render_failed=failed,
                        warning=f"Не удалось смонтировать {len(failed)} клип(ов) — подробности в журнале (data/logs).")
        # лучшие короткие фразы новых клипов — в архив для подборок «Лучшее» (не удаляется автоочисткой)
        new = [x for x in clips if x.get("candidate") in {c["cand"] for c in todo}]
        if new:
            try:
                from . import highlights
                if highlights.enabled(self.cfg):
                    self.progress("render", 0.99, "Claude выбирает лучшие короткие фразы для подборок")
                    usage = self.state.get("director_usage") or {}
                    highlights.extract(self.dir, new, self.cfg, usage,
                                       streamer=self.cfg["llm"].get("streamer_login") or self.state.get("streamer") or "")
                    self.update(director_usage=usage)
            except Exception as e:
                log.warning("Лучшие фразы не выбраны: %s", e)
        return clips

    def _prepare_stream_cam(self, chosen: list, srcs: dict) -> None:
        """Ищет вебку стримера сразу по всем клипам и запоминает её (для перемонтажа и следующих стримов)."""
        from . import streamers
        from .render import sample_for_cam, stream_cam
        r = self.cfg["render"]
        if r.get("cam_rect") or r.get("layout") not in ("auto", "split", "cam"):
            return
        self.progress("render", 0.3, "Ищу вебку стримера по всем клипам")
        samples = []
        for c in chosen:
            if c["cand"] in srcs:
                src, off = srcs[c["cand"]]
                try:
                    samples.append(sample_for_cam(src, off, c["end"] - c["start"]))
                except Exception as e:
                    log.warning("Кадры для поиска вебки: %s", e)
        prev = read_json(self.dir / "stream_cam.json")
        prior = prev or r.get("cam_prior")
        cam = stream_cam(samples, prior) if samples else None
        if cam is None and prev:
            cam = prev
        if cam and cam.get("box") and not cam.get("verified"):
            # ИИ-режиссёр один раз смотрит на рамку вебки: не захватывает ли чат/экран, не обрезает ли камеру
            from . import director
            pr = r.get("cam_prior") or {}
            same_as_verified = pr.get("verified") and pr.get("box") and \
                max(abs(a - b) for a, b in zip(pr["box"], cam["box"])) < 0.01
            if same_as_verified:
                cam["verified"] = True
            elif director.enabled(self.cfg) and (self.cfg.get("director") or {}).get("verify_cam", True):
                frames = [fr for fs, _ in samples for fr in fs[:: max(1, len(fs) // 3)]][:48]
                try:
                    self.progress("render", 0.32, "Claude проверяет рамку вебки")
                    usage = self.state.get("director_usage") or {}
                    res = director.verify_cam(frames, cam["box"], self.cfg, self.dir, usage)
                    self.update(director_usage=usage)
                    if res["box"]:
                        log.info("Рамка вебки исправлена: %s → %s", cam["box"], res["box"])
                        cam.update(box=res["box"], corrected=True)
                    cam["verified"] = bool(res["ok"] or res["box"])
                except Exception as e:
                    log.warning("Проверка рамки вебки не удалась: %s", e)
        if cam:
            r["stream_cam"] = cam
            write_json(self.dir / "stream_cam.json", cam)
            login = self.cfg["llm"].get("streamer_login")
            if login and (cam.get("verified") or cam.get("clips", 0) >= 3 or (cam.get("clips", 0) >= 2 and cam.get("box"))):
                streamers.set_cam_auto(login, cam)


def subtract_intervals(keep: list, cuts: list) -> list[tuple[float, float]]:
    """keep минус cuts (оба — списки (a, b))."""
    out = []
    for a, b in keep:
        segs = [(a, b)]
        for x, y in sorted(cuts):
            nxt = []
            for p, q in segs:
                if y <= p or x >= q:
                    nxt.append((p, q))
                else:
                    if x > p:
                        nxt.append((p, x))
                    if y < q:
                        nxt.append((y, q))
            segs = nxt
        out.extend(sg for sg in segs if sg[1] - sg[0] > 0.05)
    return out


def finalize_bounds(c: dict, s: float, e: float, words: list[dict], cfg: dict, long: bool = False) -> tuple[float, float]:
    """Привязывает границы к словам: не режем слово пополам, оставляем хвост реакции, соблюдаем длину."""
    cl = cfg["clips"]
    mn = float(cl["min_duration"])
    mx = float(cl["max_duration"])
    from .llm import AFTER_SEC, BEFORE_SEC
    lo, hi = max(0.0, c["start"] - BEFORE_SEC), c["end"] + AFTER_SEC + 3
    # не тянем клип через обрыв стрима (длинную тишину после окна)
    for w0, w1 in zip(words, words[1:]):
        if c["end"] <= w0["e"] <= hi and w1["s"] - w0["e"] > 25:
            hi = max(c["end"] + 3, w0["e"] + 1.0)
            break
    if not (e > s):
        e = min(hi, c["peak"] + 6)
        s = e - cl["target_duration"]
    s, e = max(lo, s), min(hi, e)
    ws = [w for w in words if w["e"] > lo - 1 and w["s"] < hi + 1]

    # начало: если попали внутрь слова — к его началу; иначе к ближайшему следующему слову
    for w in ws:
        if w["s"] < s < w["e"]:
            s = w["s"] - 0.15
            break
    else:
        nxt = [w for w in ws if s - 0.3 <= w["s"] <= s + 1.5]
        if nxt:
            s = nxt[0]["s"] - 0.2

    # конец: дослушиваем слово + хвост реакции 0.7 с, но не обрезая следующее слово
    for i, w in enumerate(ws):
        if w["s"] < e < w["e"] + 0.05:
            e = w["e"]
            break
    after = [w for w in ws if w["s"] >= e]
    tail = 0.7
    if after and after[0]["s"] - e < tail:
        tail = max(0.1, after[0]["s"] - e - 0.05)
    e = e + tail

    if e - s < mn:
        e = min(hi, s + mn)
        if e - s < mn:
            s = max(lo - 5, e - mn)
    if e - s > mx:
        s = e - mx
        nxt = [w for w in ws if w["s"] >= s]
        if nxt and nxt[0]["s"] - s < 2:
            s = nxt[0]["s"] - 0.15
    return round(max(0.0, s), 2), round(e, 2)
