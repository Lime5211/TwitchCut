"""Смысловой просмотр всего стрима: Claude читает полную расшифровку и находит истории, новости,
мнения, интересную информацию — даже там, где чат молчал и никто не кричал.

Это второй источник кандидатов (первый — всплески чата и звука в signals.py).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .llm import CATEGORIES, llm_call
from .util import ProgressFn, TwitchCutError, fmt_time, log, noop_progress, read_json, write_json

SCAN_KINDS = ["story", "news", "info", "hot_take", "emotional", "drama", "funny", "chat", "fail", "epic", "other"]

SCAN_SYSTEM = """Ты — продюсер TikTok-аккаунта с нарезками Twitch-стримов. Тебе дан кусок расшифровки стрима с таймкодами ([мм:сс], после часа — [ч:мм:сс]) и пометками об активности чата. Сегодня {today}.

Задача: найти ВСЕ фрагменты, из которых получится самостоятельный короткий ролик, интересный людям, которые НЕ знают стримера. Лучше вернуть лишний кандидат, чем пропустить хороший: потом каждый оценят отдельно.

СМЕШНОЕ — самый сильный тип у этих каналов (короткие смешные ролики набирают больше всего просмотров), и его легче всего пропустить: ты не слышишь интонацию и не видишь лицо. Пометки «↑ чат …» стоят ПОСЛЕ строк, на которые отреагировали зрители: если там смех («смех N%», «)», «ахаха»), ищи в строках прямо перед пометкой, над чем смеются, — это кандидат, даже если текст выглядит обычным. Смешное: абсурдная фраза или сравнение, донат/вопрос зрителя и неожиданный ответ, реакция на дикое видео, самоирония, подкол, спор с чатом, неловкость. Такие фрагменты короткие (10–40 с).

Ищи также то, что по чату не видно:
- story — история из жизни с завязкой и концовкой и с поворотом/панчлайном (длинный описательный рассказ без поворота у этих каналов набирает меньше всего — бери из него только самую сильную часть);
- news — новости и актуальное: свежие события, слухи, анонсы, цены, релизы, скандалы, известные люди и блогеры;
- info — интересный факт, совет, неожиданная информация;
- hot_take — мнение или горячий тейк о людях, играх, деньгах, отношениях, работе, трендах;
- emotional — искренность, признание, трогательный или сильный эмоциональный момент;
- drama — конфликт, спор (в том числе с чатом), разоблачение;
- funny — шутка с понятным панчлайном, абсурд, смешной диалог, реакция на видео или донат;
- chat — смешное или острое общение с чатом/донатами;
- fail / epic — игровой провал или крутой момент, понятный без знания игры.

НЕ включай: рутинную игру без смысла, технические паузы, приветствия, чтение донатов без содержания, бессвязный крик, повторы одной темы (бери лучший кусок).

НЕ включай и то, что TikTok режет или что опасно выкладывать: реальное насилие и жестокие аварии, оружие, смерть, наркотики, 18+.
Если посреди фрагмента «тишина … упал стрим» и после неё тема не продолжается — фрагмент не законченный, не бери его (или закончи до обрыва, если смысл уже есть).

Для каждого фрагмента: start и end — абсолютные таймкоды в секундах стрима (смотри [мм:сс] или [ч:мм:сс] перед строками; переведи в секунды), длина — сколько нужно, чтобы фрагмент был законченным (от 10 с до 5 минут — сколько нужно самому моменту, без подгонки под шаблон; обычно хватает 20–75 с самой сути): от начала завязки до развязки/вывода, без воды. ГЛАВНОЕ — не обрывать: end ставь только ПОСЛЕ того, как история/мысль досказана (развязка, вывод, реакция). Клипы, которые обрываются на самом интересном, автор отклоняет чаще всего. Начало — там, где начинается суть (крючок или минимально нужный контекст), а не долгая подводка к ней. Описательный рассказ без поворота, панчлайна, конфликта или полезного вывода — не бери (автор такое отклоняет как скучное). topic — о чём фрагмент, 4–10 слов. why — почему зайдёт в TikTok, одно предложение. score — предварительная оценка 0–100 (85+ сильно, 70+ хорошо, ниже 55 не возвращай).
Верни не больше {limit} лучших фрагментов; если в куске есть смешные моменты — среди них должны быть и они. Если ничего достойного — пустой список.
{streamer_context}"""

SCAN_TOOL = {
    "name": "submit_segments",
    "description": "Найденные интересные фрагменты стрима",
    "input_schema": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "kind": {"type": "string", "enum": SCAN_KINDS},
                        "topic": {"type": "string"},
                        "why": {"type": "string"},
                        "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["start", "end", "kind", "topic", "why", "score"],
                },
            }
        },
        "required": ["segments"],
    },
}

SCAN_JSON = """

ФОРМАТ ОТВЕТА: строго один JSON-объект без пояснений и markdown:
{"segments":[{"start":1234.5,"end":1298.0,"kind":"story","topic":"...","why":"...","score":78}]}
kind — одно из: """ + ", ".join(SCAN_KINDS) + "."


def _hms(t: float) -> str:
    t = int(t)
    if t < 3600:
        return f"{t // 60}:{t % 60:02d}"  # [мм:сс] — короче, меньше токенов на каждую строку
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


_FILLER = None


def _is_filler(w: str) -> bool:
    """Слова-паразиты, которые не несут смысла для поиска момента (эээ, ммм) — в просмотре не нужны."""
    global _FILLER
    if _FILLER is None:
        import re
        _FILLER = re.compile(r"^(э+|м+|а-а+|э-э+|хм+|ммм+|ээ+м*)[.,!?…]*$", re.I)
    return bool(_FILLER.match(w.strip()))


BUCKET = 30.0  # пометки о чате — по 30-секундным отрезкам


def chat_minutes(chat: list, duration: float, delay: float) -> dict[int, str]:
    """Пометки «чат оживился» по 30-секундным отрезкам — чтобы Claude видел реакцию зрителей рядом с текстом.
    В пометке: во сколько раз чат активнее обычного, доля смеха/шока и 1–2 типичных сообщения: по ним видно,
    над чем смеются (интонацию и картинку Claude не видит, а смех чата — видит)."""
    import re
    from collections import Counter
    from .lexicon import classify_message, is_noise
    n = int(duration // BUCKET) + 1
    cnt = np.zeros(n)
    cats: list[dict] = [dict() for _ in range(n)]
    msgs: list[list[str]] = [[] for _ in range(n)]
    for t, user, text in chat or []:
        m = int(max(0, t - delay) // BUCKET)
        if m >= n or is_noise(user, text):
            continue
        cnt[m] += 1
        for c, w in classify_message(text).items():
            cats[m][c] = cats[m].get(c, 0) + w
        msgs[m].append(str(text))
    if not cnt.any():
        return {}
    names = {"funny": "смех", "hype": "хайп", "shock": "шок", "fail": "провал", "cringe": "кринж", "wholesome": "мило"}
    # «обычная» активность — по соседним 10 минутам: в разные части стрима чат пишет по-разному
    # (за игрой молчит, на донатах бурлит), и всплеск надо мерить относительно местного фона
    half = int(300 // BUCKET)
    shares = np.array([cats[m].get("funny", 0) / cnt[m] if cnt[m] else 0.0 for m in range(n)])
    base_laugh = float(np.median(shares[cnt > 0])) if (cnt > 0).any() else 0.1
    out = {}
    for m in range(n):
        if not cnt[m]:
            continue
        win = cnt[max(0, m - half):m + half + 1]
        med = max(3.0, float(np.median(win[win > 0])) if (win > 0).any() else 3.0)
        ratio = cnt[m] / med
        laugh = shares[m]
        # всплеск сообщений — или заметно больше смеха, чем обычно у этого чата (смеяться могут и при
        # обычном темпе чата: так было на клипе про «белку-летягу», который набрал больше всех просмотров)
        if ratio < 1.6 and not (laugh >= max(0.22, 2.2 * base_laugh) and ratio >= 0.6):
            continue
        top = sorted(cats[m].items(), key=lambda kv: -kv[1])[:2]
        mood = ", ".join(f"{names.get(k, k)} {min(99, int(100 * v / cnt[m]))}%" for k, v in top if v / cnt[m] >= 0.1)
        # типичные сообщения: самое частое (мем/паста) и одно «словами» — без одних эмоутов
        norm = Counter(re.sub(r"\s+", " ", x.strip().lower())[:60] for x in msgs[m])
        sample = [k for k, c in norm.most_common(3) if c >= 2][:1]
        words = [x for x in msgs[m] if len(re.findall(r"[а-яёa-z]{3,}", x.lower())) >= 2 and len(x) <= 70]
        if words:
            sample.append(max(words, key=len) if len(words) < 3 else words[len(words) // 2])
        txt = f"(чат ×{ratio:.1f}" + (f": {mood}" if mood else "")
        if sample:
            txt += "; пишут: " + ", ".join("«" + x.strip()[:60] + "»" for x in sample[:2])
        out[m] = txt + ")"
    return out


def transcript_text(words: list[dict], start: float, end: float, chat_marks: dict[int, str]) -> str:
    """Компактная расшифровка: строки по предложениям/паузам с абсолютными таймкодами. Пометки о чате
    («↑ чат …») ставятся ПОСЛЕ строк того 30-секундного отрезка, на который зрители отреагировали
    (задержка чата уже учтена)."""
    lines: list[str] = []
    cur: list[str] = []
    cur_t = None
    last_e = None
    done_b = int(start // BUCKET) - 1  # до какого отрезка пометки уже выведены

    def flush_marks(upto: int) -> None:
        nonlocal done_b
        for k in range(done_b + 1, upto + 1):
            if k in chat_marks:
                lines.append("↑ " + chat_marks[k])
        done_b = max(done_b, upto)

    for w in words:
        if w["s"] < start or w["s"] >= end:
            continue
        if _is_filler(w["w"]):
            continue
        # строки подлиннее (до ~30 слов, разрыв на паузе > 1.5 с или конце длинной фразы): таймкод каждые
        # 8–12 с — этого хватает, чтобы найти фрагмент, а точные границы потом ставит оценка
        if cur and (w["s"] - last_e > 1.5 or len(cur) >= 30 or (cur[-1].endswith((".", "!", "?")) and len(cur) >= 14)):
            lines.append(f"[{_hms(cur_t)}] " + " ".join(cur))
            cur = []
        if last_e is not None and w["s"] - last_e > 25:
            flush_marks(int(last_e // BUCKET))
            lines.append(f"[{_hms(last_e)}] (тишина {w['s'] - last_e:.0f} с — пропал звук, перерыв или упал стрим)")
        if not cur:
            cur_t = w["s"]
            flush_marks(int(cur_t // BUCKET) - 1)
        cur.append(w["w"])
        last_e = w["e"]
    if cur:
        lines.append(f"[{_hms(cur_t)}] " + " ".join(cur))
    flush_marks(int(min(end, (last_e or start) + BUCKET) // BUCKET))
    return "\n".join(lines)


def scan_stream(meta: dict, tr: dict, chat: list, cfg: dict, job_dir: Path, usage: dict,
                progress: ProgressFn = noop_progress, start: float = 0.0, end: float | None = None) -> list[dict]:
    """Просматривает расшифровку кусками и возвращает найденные фрагменты (кэш по кускам)."""
    sc = cfg["scan"]
    dur = float(end if end is not None else meta.get("duration") or 0)
    words = tr["words"]
    if not words or dur <= 0:
        return []
    chunk = float(sc["chunk_minutes"]) * 60
    overlap = 150.0  # запас на стыке кусков, чтобы история не обрывалась на границе
    marks = chat_minutes(chat, dur, float(cfg["candidates"]["chat_delay_sec"]))
    parts_dir = job_dir / "scan_parts"
    parts_dir.mkdir(exist_ok=True)
    import time as _t
    ctx = short_context(cfg["llm"].get("streamer_context") or "", int(sc.get("context_chars", 700)))
    # куски одинаковой длины: вместо «25 + 25 + хвост 10 мин» (три запроса с одним и тем же системным
    # промптом) — два по 32 мин; так меньше повторов и стыков
    span = max(1.0, dur - start)
    n_chunks = max(1, int(np.ceil((span - 60) / chunk)))
    chunk = span / n_chunks
    limit = max(3, int(round(sc["per_hour"] * chunk / 3600)))
    system = SCAN_SYSTEM.format(today=_t.strftime("%d.%m.%Y"), limit=limit,
                                streamer_context=("\nО СТРИМЕРЕ:\n" + ctx) if ctx else "")
    from .feedback import prompt_block
    fb = prompt_block(cfg["llm"].get("streamer_login") or None, max_pos=4, max_neg=4, quote_chars=90)
    if fb:
        system += "\n\n" + fb
    starts = [start + k * chunk for k in range(n_chunks)]
    found: list[dict] = []
    model = sc.get("model") or cfg["llm"].get("cli_model") or "sonnet"
    for i, s in enumerate(starts):
        e = min(dur, s + chunk + overlap)
        part = parts_dir / f"scan_{int(s):06d}_{int(e):06d}.json"
        data = read_json(part)
        if data is None:
            text = transcript_text(words, s, e, marks)
            if len(text) < 200:
                data = {"segments": []}
            else:
                progress("scan", i / len(starts),
                         f"Claude читает стрим: {fmt_time(s)}–{fmt_time(e)} ({i + 1} из {len(starts)})")
                user = (f"Стрим: «{meta.get('title', '')}», канал: {meta.get('channel', '')}. "
                        f"Кусок {_hms(s)}–{_hms(e)}.\n\n{text}")
                data = None
                for attempt in range(2):
                    try:
                        data = llm_call(system, user, cfg, usage, job_dir, tool=SCAN_TOOL,
                                        json_instruction=SCAN_JSON, cli_model=model)
                        break
                    except TwitchCutError as ex:
                        msg = str(ex)
                        if "лимит" in msg.lower() or "авториз" in msg.lower():
                            raise  # лимит/вход — повтор бесполезен
                        log.warning("Просмотр куска %s–%s не удался (%s)%s", fmt_time(s), fmt_time(min(dur, s + chunk)),
                                    msg, ", повторяю" if attempt == 0 else ", пропускаю кусок")
                if data is None:
                    failed = usage.setdefault("scan_failed", [])
                    failed.append(f"{fmt_time(s)}–{fmt_time(e)}")
                    continue  # один неудачный кусок не отменяет весь поиск
                write_json(part, data)
        for seg in data.get("segments") or []:
            try:
                a, b = float(seg["start"]), float(seg["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if b <= a or a < s - 120 or b > e + 120:
                continue
            found.append({
                "start": max(0.0, a), "end": min(dur, b), "kind": seg.get("kind") if seg.get("kind") in CATEGORIES else "other",
                "topic": str(seg.get("topic") or "")[:120], "why": str(seg.get("why") or "")[:300],
                "score": int(max(0, min(100, float(seg.get("score") or 0)))),
            })
    # убираем дубли с перекрывающихся кусков
    found.sort(key=lambda x: -x["score"])
    uniq: list[dict] = []
    for f in found:
        if any(_overlap(f, u) > 0.5 for u in uniq):
            continue
        uniq.append(f)
    uniq.sort(key=lambda x: x["start"])
    progress("scan", 1.0, f"Найдено по смыслу: {len(uniq)}")
    log.info("Смысловой просмотр: %d фрагментов", len(uniq))
    return uniq


def short_context(ctx: str, max_chars: int) -> str:
    """Описание стримера для просмотра — коротко: имя, как его называют, о чём стримы. Без markdown-разметки
    (звёздочки и решётки — лишние токены). Полностью описание идёт в оценку, где пишутся названия."""
    import re
    t = re.sub(r"[*#`_]+", "", ctx or "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    cut = cut[:max(cut.rfind("\n"), cut.rfind(". ") + 1, int(max_chars * 0.6))]
    return cut.strip() + " …"


def _overlap(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / max(1e-6, min(a["end"] - a["start"], b["end"] - b["start"]))


def merge_candidates(signal_cands: list[dict], speech: list[dict], timeline: dict, cfg: dict) -> list[dict]:
    """Объединяет кандидатов из сигналов (чат/звук) и из смыслового просмотра."""
    pad = 8.0
    out: list[dict] = []
    used = set()
    for k, sp in enumerate(speech, 1):
        c = {"id": f"s{k:02d}", "source": "speech", "start": max(0.0, sp["start"] - pad), "end": sp["end"] + pad,
             "peak": sp["end"], "topic": sp["topic"], "kind": sp["kind"], "why": sp["why"],
             "pre_score": sp["score"], "signal": 0.0, "signal_norm": 0.0}
        # если рядом есть всплеск чата/звука — объединяем подсказки
        for j, sc in enumerate(signal_cands):
            if j in used:
                continue
            if sc["peak"] >= sp["start"] - 10 and sc["peak"] <= sp["end"] + 15:
                used.add(j)
                c["source"] = "speech+signals"
                for key in ("chat_ratio", "reaction", "loud_delta_db", "viewer_clips", "chat", "signal", "signal_norm"):
                    if key in sc:
                        c[key] = sc[key]
                c["start"] = min(c["start"], sc["start"] + 30)  # не раздуваем окно слишком сильно
                c["end"] = max(c["end"], min(sc["end"], sc["peak"] + 12))
        if timeline.get("score") and not c.get("signal"):
            step = timeline["step"]
            i0, i1 = int(c["start"] // step), int(c["end"] // step) + 1
            seg = timeline["score"][i0:i1] or [0]
            c["signal"] = float(max(seg))
        out.append(c)
    for j, sc in enumerate(signal_cands):
        if j not in used:
            out.append({**sc, "source": "signals"})
    # ограничиваем число кандидатов для финальной оценки
    mx = int(cfg["scan"]["max_candidates"])
    if len(out) > mx:
        def prio(c):
            if c["source"].startswith("speech"):
                return c.get("pre_score", 50) + 20 * c.get("signal_norm", 0)
            return 45 + 50 * c.get("signal_norm", 0)
        out = sorted(out, key=prio, reverse=True)[:mx]
    # нормированная сила сигнала для всех
    sigs = [c.get("signal", 0.0) for c in out]
    if sigs:
        lo, hi = min(sigs), max(sigs)
        for c in out:
            c["signal_norm"] = round((c.get("signal", 0.0) - lo) / (hi - lo), 3) if hi > lo else 0.5
    out.sort(key=lambda c: c["start"])
    return out
