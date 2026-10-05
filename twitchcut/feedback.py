"""Обратная связь по клипам: «не подходит» (с причиной) и «выложил» (с просмотрами).

Эти отметки становятся примерами в промптах Claude: что у этого канала реально залетело,
что зашло слабо и что автор отбраковал. Так система постепенно подстраивается под ваш вкус
и под то, что набирает просмотры именно у вас.
"""
from __future__ import annotations

import statistics
import threading
import time
from pathlib import Path

from .util import ROOT_DIR, log, read_json, write_json

PATH = ROOT_DIR / "data" / "feedback.json"
_lock = threading.Lock()

REASONS = {
    "boring": "скучно", "context": "непонятно без контекста", "cut": "плохо обрезано", "not_funny": "не смешно",
    "repeat": "повтор / уже было", "quality": "плохая картинка или звук", "risky": "нельзя выкладывать",
    "other": "другое",
}
CAT_RU = {"funny": "смешное", "story": "история", "news": "новость", "info": "интересное", "hot_take": "мнение",
          "emotional": "эмоции", "drama": "драма", "fail": "фейл", "epic": "эпик", "rage": "ярость",
          "scare": "испуг", "cringe": "кринж", "chat": "чат", "wholesome": "мило", "other": "другое"}


def load() -> list[dict]:
    return read_json(PATH) or []


def for_job(job_id: str) -> dict[str, dict]:
    return {f["file"]: f for f in load() if f.get("job") == job_id}


def upsert(entry: dict) -> dict:
    with _lock:
        items = load()
        key = f"{entry['job']}/{entry['file']}"
        cur = next((x for x in items if x.get("id") == key), {})
        rec = {**cur, **{k: v for k, v in entry.items() if v is not None}, "id": key, "updated": time.time()}
        if rec.get("status") == "posted" and not rec.get("posted_at"):
            rec["posted_at"] = time.time()
        if rec.get("status") in ("none", "", None):
            items = [x for x in items if x.get("id") != key]
        else:
            items = [x for x in items if x.get("id") != key] + [rec]
        write_json(PATH, items)
    return rec


def replace(rec: dict) -> None:
    """Перезаписать отметку целиком (используется фоновым обновлением статистики)."""
    with _lock:
        items = load()
        if not any(x.get("id") == rec["id"] for x in items):
            return  # отметку успели снять
        items = [rec if x.get("id") == rec["id"] else x for x in items]
        write_json(PATH, items)


def fetch_stats_async(rec: dict) -> None:
    """Сразу после «Выложил» со ссылкой TikTok — подтянуть просмотры в фоне."""
    from . import tiktok
    if rec.get("status") != "posted" or not tiktok.is_tiktok(rec.get("url") or ""):
        return
    threading.Thread(target=lambda: tiktok.refresh({rec["id"]}), daemon=True).start()


def _dur_bucket(d) -> str:
    d = float(d or 0)
    return "до 30 с" if d < 30 else "30–60 с" if d < 60 else "1–2 мин" if d < 120 else "2+ мин"


def _fmt_views(v) -> str:
    v = int(v or 0)
    if v >= 1_000_000:
        return f"{v / 1e6:.1f} млн".replace(".0 ", " ")
    if v >= 1000:
        return f"{v / 1000:.0f} тыс."
    return str(v)


def stats(login: str | None = None) -> dict:
    items = [f for f in load() if not login or f.get("streamer") == login]
    posted = [f for f in items if f.get("status") == "posted"]
    with_views = [f for f in posted if f.get("views") is not None]
    rejected = [f for f in items if f.get("status") == "rejected"]
    by_cat: dict[str, list] = {}
    for f in with_views:
        by_cat.setdefault(f.get("category") or "other", []).append(int(f["views"]))
    reasons: dict[str, int] = {}
    for f in rejected:
        for r in f.get("reasons") or []:
            reasons[r] = reasons.get(r, 0) + 1
    views = [int(f["views"]) for f in with_views]
    by_dur: dict[str, list] = {}
    for f in with_views:
        by_dur.setdefault(_dur_bucket(f.get("final_duration") or f.get("tt_duration")), []).append(int(f["views"]))
    eng = [(f.get("likes") or 0) / f["views"] for f in with_views if f.get("likes") is not None and f["views"] > 0]
    return {
        "posted": len(posted), "rejected": len(rejected), "with_views": len(with_views),
        "median_views": int(statistics.median(views)) if views else 0,
        "total_views": sum(views),
        "total_likes": sum(int(f.get("likes") or 0) for f in with_views),
        "total_comments": sum(int(f.get("comments") or 0) for f in with_views),
        "like_rate": round(100 * statistics.median(eng), 1) if eng else None,
        "auto": sum(1 for f in posted if f.get("stats_auto")),
        "by_category": {k: {"n": len(v), "median": int(statistics.median(v))} for k, v in by_cat.items()},
        "by_duration": {k: {"n": len(v), "median": int(statistics.median(v))} for k, v in by_dur.items()},
        "reasons": reasons,
    }


def stats_by_streamer() -> dict[str, dict]:
    """Сводка по каждому стримеру отдельно + лучший ролик канала."""
    out = {}
    for login in sorted({f.get("streamer") or "" for f in load()}):
        if not login:
            continue
        st = stats(login)
        best = max((f for f in load() if f.get("streamer") == login and f.get("status") == "posted"
                    and f.get("views") is not None), key=lambda f: f["views"], default=None)
        if best:
            st["best"] = {"title": best.get("title"), "views": best.get("views"), "url": best.get("url")}
        out[login] = st
    return out


_TECH_NOTE = __import__("re").compile(r"вебк|экран|формат|раскладк|лицо|область|звук и видео|рассинхр|блюр", __import__("re").I)


def prompt_block(login: str | None, max_pos: int = 6, max_neg: int = 10) -> str:
    """Текст для промпта: что залетело, что нет, что отклонено. Пусто, если отметок ещё нет."""
    allf = load()
    items = [f for f in allf if f.get("streamer") == login] if login else allf
    if login and not items:
        items, login = allf, None  # по этому каналу отметок ещё нет — берём общий опыт
    if not items:
        return ""
    import time as _time
    # свежие ролики (меньше ~1.5 суток) ещё набирают просмотры — по ним рано судить
    posted = [f for f in items if f.get("status") == "posted" and f.get("views") is not None
              and _time.time() - float(f.get("posted_ts") or f.get("posted_at") or f.get("updated") or 0) > 36 * 3600]
    rejected = [f for f in items if f.get("status") == "rejected"]
    lines = ["ОПЫТ ПРОШЛЫХ НАРЕЗОК" + (" ЭТОГО КАНАЛА" if login else "") +
             " — учитывай вкус автора и реальные просмотры в TikTok:"]

    def one(f: dict, extra: str) -> str:
        cat = CAT_RU.get(f.get("category") or "", f.get("category") or "")
        title = (f.get("title") or f.get("topic") or "").strip()
        text = (f.get("text") or "").strip().replace("\n", " ")
        if len(text) > 160:
            text = text[:160] + "…"
        return f"- [{cat}, {extra}] «{title}»" + (f" — «{text}»" if text else "")

    def views_txt(f: dict) -> str:
        t = _fmt_views(f["views"]) + " просм."
        if f.get("likes") and f["views"]:
            t += f", лайков {100 * f['likes'] / f['views']:.0f}%"
        if f.get("comments"):
            t += f", комм. {f['comments']}"
        if f.get("final_duration"):
            t += f", {f['final_duration']:.0f} с"
        return t

    if posted:
        med = statistics.median([int(f["views"]) for f in posted])
        good = sorted([f for f in posted if f["views"] >= med], key=lambda f: -f["views"])[:max_pos]
        bad = sorted([f for f in posted if f["views"] < med * 0.5], key=lambda f: f["views"])[:max(2, max_pos // 2)]
        if good:
            lines.append(f"Зашли лучше всего (медиана канала {_fmt_views(med)} просмотров):")
            lines += [one(f, views_txt(f)) for f in good]
        if bad:
            lines.append("Зашли слабо:")
            lines += [one(f, views_txt(f)) for f in bad]
        st = stats(login)
        cats = sorted(st["by_category"].items(), key=lambda kv: -kv[1]["median"])
        if len(cats) >= 2:
            lines.append("Медиана просмотров по типам: " + ", ".join(
                f"{CAT_RU.get(k, k)} {_fmt_views(v['median'])} ({v['n']})" for k, v in cats))
        durs = sorted(st["by_duration"].items(), key=lambda kv: -kv[1]["median"])
        if len(durs) >= 2 and sum(v["n"] for _, v in durs) >= 4:
            lines.append("Медиана просмотров по длине клипа: " + ", ".join(
                f"{k} {_fmt_views(v['median'])} ({v['n']})" for k, v in durs))
    if rejected:
        lines.append("Автор отклонил (не стал выкладывать) — избегай похожего:")
        for f in sorted(rejected, key=lambda f: -f.get("updated", 0))[:max_neg * 2]:
            if sum(1 for l in lines if l.startswith("- [") and "причина:" in l) >= max_neg:
                break
            why = ", ".join(REASONS.get(r, r) for r in (f.get("reasons") or []))
            note = (f.get("note") or "").strip()
            # заметки про картинку (вебка, раскладка, звук) Claude не помогут — их учитывает монтаж
            if note and not _TECH_NOTE.search(note):
                why = (why + "; " if why else "") + note[:220]
            elif not why and note:
                continue
            lines.append(one(f, "причина: " + (why or "не понравилось")))
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


# ------------------------------------------------------------ монтажные замечания
def edit_notes(login: str | None, max_n: int = 14) -> str:
    """Замечания автора про картинку (вебка, экран, формат, переключения) — для ИИ-режиссёра.
    Сначала по этому каналу, затем общие (вкус к монтажу у автора один на все каналы)."""
    from . import streamers
    items = [f for f in load() if f.get("note") and _TECH_NOTE.search(f["note"])]
    items.sort(key=lambda f: (f.get("streamer") != login, -f.get("updated", 0)))
    lines = []
    st = streamers.get(login) if login else None
    if st and st.get("learned_edit"):
        lines.append(st["learned_edit"].strip())
    if items:
        lines.append("Замечания автора к прошлым клипам:")
        for f in items[:max_n]:
            lay = f" (было: {f['layout']})" if f.get("layout") else ""
            lines.append(f"- «{(f.get('title') or '')[:70]}»{lay}: {f['note'][:220]}")
    return "\n".join(lines)


# ------------------------------------------------------------ вкус канала
LEARN_SYSTEM = """Ты анализируешь, какие клипы со стримов заходят у автора TikTok-аккаунта нарезок, а какие он отбраковывает.
По его отметкам («выложил» с реальными просмотрами/лайками и «не подходит» с причинами и комментариями) составь
КОРОТКИЙ профиль вкуса для этого канала — его будут читать модели, которые ищут и монтируют клипы.

taste — 5–9 пунктов «- …»: какие темы/типы моментов у ЭТОГО стримера заходят (с опорой на просмотры), что брать
нельзя или не стоит (скучные типы, повторяющиеся рубрики, которые автор отклоняет), какая длина лучше, на что
обращать внимание в начале и конце клипа. Конкретно, с примерами тем, без воды.
edit — 3–7 пунктов «- …» про картинку и монтаж (вебка/экран/когда переключать/что показывать крупно/темп),
только то, что следует из замечаний автора. Если замечаний про монтаж нет — пустая строка.
Не выдумывай того, чего нет в данных."""

LEARN_JSON = """

ФОРМАТ ОТВЕТА: строго один JSON-объект: {"taste":"- ...\\n- ...","edit":"- ...\\n- ..."}"""

LEARN_TOOL = {"name": "submit_profile", "description": "Профиль вкуса канала", "input_schema": {
    "type": "object", "properties": {"taste": {"type": "string"}, "edit": {"type": "string"}}, "required": ["taste"]}}


def learn_profile(login: str, cfg: dict, workdir, force: bool = False) -> dict | None:
    """Обновляет «вкус канала» по отметкам автора (если с прошлого раза появилось ≥3 новых отметки)."""
    from . import streamers
    from .llm import llm_call
    st = streamers.get(login)
    if not st:
        return None
    items = [f for f in load() if f.get("streamer") == login and f.get("status") in ("posted", "rejected")]
    if len(items) < 4:
        return None
    if not force and len(items) - int(st.get("learned_n") or 0) < 3:
        return None
    rows = []
    for f in sorted(items, key=lambda f: -f.get("updated", 0))[:60]:
        dur = f.get("final_duration") or f.get("duration")
        base = f"«{(f.get('title') or '')[:80]}» [{CAT_RU.get(f.get('category') or '', f.get('category') or '')}, {dur or '?'} с]"
        if f["status"] == "posted":
            v = f.get("views")
            extra = f"выложен: {v if v is not None else '?'} просм." + (f", {f['likes']} лайков" if f.get("likes") else "")
            if time.time() - float(f.get("posted_ts") or f.get("posted_at") or f.get("updated") or 0) < 36 * 3600:
                extra += " (выложен меньше 1.5 суток назад — ещё набирает)"
        else:
            why = ", ".join(REASONS.get(r, r) for r in (f.get("reasons") or []))
            extra = "отклонён: " + (why or "") + (("; " + f["note"][:200]) if f.get("note") else "")
        rows.append(f"- {base} — {extra}")
    user = (f"Канал: {st.get('name') or login}.\nОписание стримера: {(st.get('description') or '')[:800]}\n\n"
            f"Отметки автора (новые сверху):\n" + "\n".join(rows))
    usage: dict = {}
    data = llm_call(LEARN_SYSTEM, user, cfg, usage, Path(workdir), tool=LEARN_TOOL, json_instruction=LEARN_JSON)
    taste = str(data.get("taste") or "").strip()[:1800]
    edit = str(data.get("edit") or "").strip()[:1200]
    if not taste:
        return None
    streamers.set_fields(login, learned_taste=taste, learned_edit=edit, learned_n=len(items), learned_at=time.time())
    log.info("Вкус канала %s обновлён по %d отметкам", login, len(items))
    return {"taste": taste, "edit": edit}
