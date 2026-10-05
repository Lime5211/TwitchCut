"""Архив «Лучшее»: короткие сильные моменты (2–12 с) из всех клипов всех стримов — для подборок
«Топ моментов» за неделю/месяц/всё время.

Как это работает:
- после монтажа клипов Claude читает их расшифровку и выбирает 0–2 самых сильных КОРОТКИХ кусочка из каждого:
  фраза, которая смешно звучит вырванной из контекста, самоирония/подкол стримера, дерзкая или абсурдная
  цитата, резкая реакция, панчлайн;
- эти кусочки вырезаются из готового клипа (с субтитрами и запиканным матом) и сохраняются в data/archive —
  автоочистка их не трогает. Один кусочек ≈ 2–5 МБ; у архива есть предел размера, при переполнении
  удаляются самые слабые и старые моменты, а моменты из клипов, набравших много просмотров, — в последнюю
  очередь;
- подборка собирается из лучших моментов стримера за период: сначала то, что набрало больше просмотров в
  TikTok (относительно обычных просмотров канала), затем оценка Claude.
"""
from __future__ import annotations

import hashlib
import statistics
import threading
import time
from pathlib import Path

from .util import ROOT_DIR, TwitchCutError, ffmpeg_bin, log, read_json, run, write_json

ARCHIVE = ROOT_DIR / "data" / "archive"
INDEX = ARCHIVE / "index.json"
_lock = threading.RLock()

MOMENTS_SYSTEM = """Ты — монтажёр TikTok-аккаунта, который делает подборки «лучшие моменты стримера». Тебе даны расшифровки
уже готовых клипов (таймкоды [сек] — от начала клипа). Из каждого клипа выбери 0–2 самых сильных КОРОТКИХ кусочка
(2–12 секунд), которые зайдут в подборку сами по себе, без остального клипа:
- фраза, которая смешно или дико звучит, вырванная из контекста;
- подкол или «унижение» стримера: чат/донат/собеседник его приложил, он сам себя уронил, самоирония, неловкость;
- дерзкая, абсурдная или крылатая цитата;
- резкая реакция (крик, ступор, истерический смех) с понятным поводом прямо в кусочке;
- панчлайн шутки, если он понятен без подводки.
Кусочек должен быть понятен за 2 секунды БЕЗ предыстории, начинаться с начала фразы и заканчиваться после неё
(плюс секунда реакции). Не бери длинные объяснения, истории, то, что требует контекста, и опасные темы.
Если в клипе такого нет — ничего из него не бери. quote — сама фраза (коротко), score 0–100 — насколько это
зайдёт в подборке (85+ — бомба, 70+ — хорошо, ниже 60 не возвращай).
{taste}"""

MOMENTS_TOOL = {"name": "submit_moments", "description": "Лучшие короткие моменты", "input_schema": {
    "type": "object", "properties": {"moments": {"type": "array", "items": {"type": "object", "properties": {
        "clip": {"type": "integer"}, "start": {"type": "number"}, "end": {"type": "number"},
        "quote": {"type": "string"}, "kind": {"type": "string"}, "score": {"type": "integer"}},
        "required": ["clip", "start", "end", "quote", "score"]}}}, "required": ["moments"]}}

MOMENTS_JSON = """

ФОРМАТ ОТВЕТА: строго один JSON-объект без пояснений:
{"moments":[{"clip":3,"start":12.4,"end":19.0,"quote":"я гений, просто никто не понял","kind":"quote","score":82}]}"""


def enabled(cfg: dict) -> bool:
    h = cfg.get("highlights") or {}
    return bool(h.get("enabled", True)) and cfg["llm"]["backend"] in ("api", "claude-cli")


def load() -> list[dict]:
    return read_json(INDEX) or []


def _save(items: list[dict]) -> None:
    write_json(INDEX, items)


def _clip_lines(words: list) -> list[str]:
    out, cur, t0, last = [], [], None, None
    for w, s, e in words:
        if cur and (s - last > 0.7 or len(cur) >= 12 or cur[-1].endswith((".", "!", "?"))):
            out.append(f"[{t0:.1f}] " + " ".join(cur))
            cur = []
        if not cur:
            t0 = s
        cur.append(w)
        last = e
    if cur:
        out.append(f"[{t0:.1f}] " + " ".join(cur))
    return out


def _words_for(clip: dict, path: Path, cfg: dict) -> list:
    """Слова клипа в его итоговой шкале. У старых клипов их нет — распознаём речь самого клипа."""
    if clip.get("words_final"):
        return clip["words_final"]
    try:
        from .transcribe import Transcriber
        from .util import ffprobe_video
        dur = float(clip.get("final_duration") or 0) or 60.0
        words, _ = Transcriber(cfg).transcribe_range(path, 0.0, dur)
        return [[w["w"], w["s"], w["e"]] for w in words]
    except Exception as e:
        log.info("Лучшие фразы: не удалось распознать речь клипа %s: %s", path.name, e)
        return []


def _cut(src: Path, a: float, b: float, out: Path, cfg: dict) -> None:
    from .render import VCODEC_ARGS, X264_EXTRA, pick_encoder
    enc = pick_encoder(cfg)
    vcodec = VCODEC_ARGS.get(enc) or ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", *X264_EXTRA]
    d = b - a
    vf = f"fade=t=in:st=0:d=0.12,fade=t=out:st={max(0.0, d - 0.15):.2f}:d=0.15"
    af = f"afade=t=in:d=0.08,afade=t=out:st={max(0.0, d - 0.15):.2f}:d=0.15"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    cmd = [ffmpeg_bin(), "-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-i", str(src),
           "-vf", vf, "-af", af, *vcodec, "-r", "30", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)]
    try:
        run(cmd)
    except Exception:
        i = cmd.index(vcodec[0])
        cmd[i:i + len(vcodec)] = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", *X264_EXTRA]
        run(cmd)
    tmp.replace(out)
    run([ffmpeg_bin(), "-y", "-v", "error", "-ss", f"{min(0.8, d / 2):.2f}", "-i", str(out), "-frames:v", "1",
         "-vf", "scale=240:-2", str(out.with_suffix(".jpg"))], check=False)


def extract(job_dir: Path, clips: list[dict], cfg: dict, usage: dict, streamer: str = "") -> list[dict]:
    """Выбирает лучшие короткие моменты из клипов задачи и сохраняет их в архив."""
    from .llm import llm_call
    blocks, by_n = [], {}
    for c in clips:
        p = job_dir / c["file"]
        if not p.exists():
            continue
        words = _words_for(c, p, cfg)
        if len(words) < 4:
            continue
        by_n[int(c["n"])] = (c, p)
        blocks.append(f"### Клип {c['n']} «{c.get('title') or ''}» (длина {c.get('final_duration') or '?'} с)\n"
                      + "\n".join(_clip_lines(words)))
    if not blocks:
        return []
    taste = ""
    try:
        from . import streamers
        st = streamers.get(streamer) if streamer else None
        if st and st.get("learned_taste"):
            taste = "\nЧТО ЗАХОДИТ У ЭТОГО КАНАЛА:\n" + st["learned_taste"][:1200]
        if st and st.get("description"):
            taste = f"\nО СТРИМЕРЕ: {st['description'][:600]}" + taste
    except Exception:
        pass
    data = llm_call(MOMENTS_SYSTEM.format(taste=taste), "\n\n".join(blocks), cfg, usage, job_dir,
                    tool=MOMENTS_TOOL, json_instruction=MOMENTS_JSON)
    added = []
    with _lock:
        items = load()
        have = {m["id"] for m in items}
        for m in data.get("moments") or []:
            try:
                n, a, b = int(m["clip"]), float(m["start"]), float(m["end"])
                score = int(m.get("score") or 0)
            except (KeyError, TypeError, ValueError):
                continue
            if n not in by_n or score < 60:
                continue
            c, p = by_n[n]
            dur = float(c.get("final_duration") or 0) or 9999
            a, b = max(0.0, a - 0.2), min(dur, b + 0.5)
            if not 1.5 <= b - a <= 15:
                continue
            mid = hashlib.md5(f"{job_dir.name}/{c['file']}/{a:.1f}".encode()).hexdigest()[:12]
            if mid in have:
                continue
            out = ARCHIVE / (streamer or "other") / f"{mid}.mp4"
            try:
                _cut(p, a, b, out, cfg)
            except Exception as e:
                log.warning("Лучшие фразы: не удалось вырезать %s: %s", mid, e)
                continue
            rec = {"id": mid, "streamer": streamer, "job": job_dir.name, "clip_file": c["file"], "clip_n": n,
                   "clip_title": c.get("title"), "quote": str(m.get("quote") or "")[:160], "kind": m.get("kind"),
                   "score": score, "dur": round(b - a, 1), "file": str(out.relative_to(ROOT_DIR)).replace("\\", "/"),
                   "created": time.time()}
            items.append(rec)
            added.append(rec)
        _save(items)
    log.info("Лучшие фразы: +%d в архив (%s)", len(added), job_dir.name)
    enforce_limit(cfg)
    return added


# ------------------------------------------------------------------ оценка
def _views_index() -> dict:
    from . import feedback
    return {f"{f.get('job')}/{f.get('file')}": f for f in feedback.load()}


def rank(items: list[dict]) -> list[dict]:
    """Ценность момента: просмотры клипа-родителя относительно медианы канала + оценка Claude.
    Моменты из отклонённых клипов — ниже, но не исключаются (фраза может быть хорошей и в слабом клипе)."""
    fb = _views_index()
    med: dict[str, float] = {}
    by_streamer: dict[str, list] = {}
    for f in fb.values():
        if f.get("status") == "posted" and f.get("views"):
            by_streamer.setdefault(f.get("streamer") or "", []).append(int(f["views"]))
    for k, v in by_streamer.items():
        med[k] = max(1.0, statistics.median(v))
    out = []
    for m in items:
        f = fb.get(f"{m['job']}/{m['clip_file']}") or {}
        views = f.get("views") if f.get("status") == "posted" else None
        rel = (views / med.get(m.get("streamer") or "", 1.0)) if views else None
        value = m["score"] / 100.0
        if rel is not None:
            value = 0.4 * value + 0.6 * min(3.0, rel) / 3.0 * 1.2
        if f.get("status") == "rejected":
            value *= 0.7
        out.append({**m, "views": views, "rel_views": round(rel, 2) if rel else None, "value": round(value, 3),
                    "exists": (ROOT_DIR / m["file"]).exists()})
    out.sort(key=lambda m: -m["value"])
    return out


def size_bytes() -> int:
    return sum(p.stat().st_size for p in ARCHIVE.rglob("*") if p.is_file()) if ARCHIVE.exists() else 0


def enforce_limit(cfg: dict) -> None:
    """Держим архив в пределах highlights.max_gb: удаляем самые слабые моменты (старые — раньше)."""
    limit = float((cfg.get("highlights") or {}).get("max_gb", 3.0)) * 1e9
    with _lock:
        if size_bytes() <= limit:
            return
        ranked = rank(load())
        now = time.time()
        ranked.sort(key=lambda m: m["value"] - min(0.3, (now - m.get("created", now)) / (90 * 86400)))
        keep = {m["id"] for m in ranked}
        for m in ranked:
            if size_bytes() <= limit * 0.9:
                break
            delete(m["id"], _locked=True)
            keep.discard(m["id"])
            log.info("Архив «Лучшее» переполнен — удалён момент %s", m["id"])


def delete(mid: str, _locked: bool = False) -> None:
    with _lock:
        items = load()
        for m in [x for x in items if x["id"] == mid]:
            p = ROOT_DIR / m["file"]
            p.unlink(missing_ok=True)
            p.with_suffix(".jpg").unlink(missing_ok=True)
        _save([x for x in items if x["id"] != mid])


# ------------------------------------------------------------------ подборка
def pick(streamer: str | None, days: float | None, n: int, max_total: float, ids: list[str] | None = None) -> list[dict]:
    items = [m for m in rank(load()) if m["exists"]]
    if ids:
        sel = {i: k for k, i in enumerate(ids)}
        return sorted([m for m in items if m["id"] in sel], key=lambda m: sel[m["id"]])
    if streamer:
        items = [m for m in items if m.get("streamer") == streamer]
    if days:
        items = [m for m in items if time.time() - m.get("created", 0) <= days * 86400]
    chosen, total, seen_clips = [], 0.0, {}
    for m in items:
        key = f"{m['job']}/{m['clip_file']}"
        if seen_clips.get(key, 0) >= 1:  # не больше одного момента из одного клипа
            continue
        if total + m["dur"] > max_total:
            continue
        chosen.append(m)
        seen_clips[key] = seen_clips.get(key, 0) + 1
        total += m["dur"]
        if len(chosen) >= n:
            break
    return chosen


def build(cfg: dict, streamer: str | None, days: float | None, n: int = 5, max_total: float = 60.0,
          ids: list[str] | None = None) -> dict:
    """Подборка «Топ-N моментов»: обратный отсчёт, на каждом моменте плашка «#N»."""
    from .compilation import concat_with_numbers
    chosen = pick(streamer, days, n, max_total, ids)
    if len(chosen) < 2:
        raise TwitchCutError("Для подборки нужно хотя бы 2 момента в архиве «Лучшее» (с этим фильтром)")
    order = list(reversed(chosen))  # лучший — в конце
    stamp = time.strftime("%Y%m%d_%H%M")
    out = ARCHIVE / "compilations" / f"best_{streamer or 'all'}_{stamp}.mp4"
    concat_with_numbers([ROOT_DIR / m["file"] for m in order], out, cfg)
    info = {"file": str(out.relative_to(ROOT_DIR)).replace("\\", "/"), "thumb": str(out.with_suffix(".jpg").relative_to(ROOT_DIR)).replace("\\", "/"),
            "streamer": streamer or "", "days": days, "n": len(order),
            "duration": round(sum(m["dur"] for m in order), 1),
            "moments": [{"id": m["id"], "quote": m["quote"], "place": len(order) - i} for i, m in enumerate(order)],
            "created": time.time()}
    comps = read_json(ARCHIVE / "compilations.json") or []
    write_json(ARCHIVE / "compilations.json", comps + [info])
    log.info("Подборка «Лучшее»: %s (%d моментов, %.0f с)", out.name, len(order), info["duration"])
    return info
