"""Понимание контекста: LLM оценивает кандидатов, выбирает точные границы, пишет хук и описание.

Бэкенды:
  api        — Anthropic Messages API (requests, без доп. зависимостей)
  claude-cli — локальный Claude Code: `claude -p` (расходует лимиты подписки, только для личного использования)
  manual     — сохраняет промпт в файл, ответ вставляется вручную (из обычного чата Claude)
  none       — эвристика по сигналам, без LLM
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import requests

from .transcribe import slice_words
from .util import ProgressFn, TwitchCutError, fmt_time, log, noop_progress

CATEGORIES = ["funny", "story", "news", "info", "hot_take", "emotional", "drama", "fail", "epic", "rage",
              "scare", "cringe", "chat", "wholesome", "other"]
# «длинные» категории: им нужно время раскрыться (история, новость, мнение)
LONG_CATEGORIES = {"story", "news", "info", "hot_take", "emotional", "drama"}

# $ за 1 млн токенов (вход, выход) — для оценки стоимости в отчёте
PRICES = {"haiku": (1.0, 5.0), "sonnet": (2.0, 10.0), "opus": (4.0, 20.0)}

TITLE_RULES = """НАЗВАНИЯ ДЛЯ TIKTOK (titles — ровно 2 варианта, на языке стрима)
- Это подпись, которую видят в ленте. 30–70 символов. Понятно без контекста: КТО и ЧТО сделал/сказал/узнал.
- Разговорный живой стиль, как у популярных нарезочных аккаунтов. Имя стримера — так, как его называют зрители (см. описание стримера). 0–1 эмодзи в конце.
- Интрига без обмана: не обещай того, чего нет в клипе, не выдумывай деталей.
- Без кавычек, хэштегов, КАПСА целиком, без «Стример…» в начале, без слов «момент», «клип», «нарезка».
- Вариант 1 — суть одной фразой: кто и что сделал/сказал (для новости — сама новость). Вариант 2 — тот же смысл, но с интригой/недосказанностью, чтобы захотелось досмотреть.
- Примеры: «Тоха принял игрока с фонарём за свой фонарь» / «Это оказывается не фонарь был 😳»; «Т2х2 о том, почему ушёл с прошлой работы» / «Тоха рассказал, почему больше туда не вернётся»."""

SYSTEM_PROMPT = """Ты — продюсер популярного TikTok-аккаунта с нарезками Twitch-стримов. Ты отлично понимаешь, какие ролики набирают просмотры у людей, которые НЕ знают стримера и листают ленту. Сегодня {today}.

ГЛАВНОЕ: оценивай СМЫСЛ сказанного, а не громкость. Активность чата и крик — лишь подсказки. Многие лучшие ролики для TikTok — это истории, новости, мнения и полезная информация, во время которых чат может молчать. Громкий крик или смех без понятного повода — СЛАБЫЙ момент.

ЧТО ЗАХОДИТ В TIKTOK (по убыванию типичной силы)
1. Новости и актуальное: обсуждение свежих событий, слухов, анонсов, цен, релизов, скандалов, известных людей и блогеров, решения властей/платформ. Сильно, если тема волнует широкую аудиторию.
2. Истории из жизни с завязкой и концовкой: странные, смешные, жизненные, откровенные.
3. Мнения и горячие тейки: о людях, играх, деньгах, отношениях, работе, трендах — то, с чем хочется согласиться или поспорить в комментариях.
4. Интересное и полезное: факты, советы, неожиданная информация, «а вы знали».
5. Смешное: шутка с понятным панчлайном, абсурд, смешной диалог, реакция на видео/донат.
6. Эмоции и драма: искренность, признание, конфликт, спор с чатом, ярость с понятной причиной.
7. Игровые моменты: фейл, клатч, испуг — только если понятны без знания игры.

КРИТЕРИИ ОЦЕНКИ
- Хук: первые 1–2 секунды цепляют (сильная фраза, вопрос, неожиданность). Не «ну короче», «так, чат», тишина.
- Самостоятельность: понятно без предыдущего часа стрима. Если не хватает контекста — сдвинь начало раньше, чтобы завязка попала в клип.
- Ценность/развязка: зритель что-то узнаёт, смеётся, удивляется или хочет спорить.
- Актуальность и широта темы: интересно ли это массовому зрителю сейчас.
- Комментируемость: захотят ли написать комментарий или переслать другу.

КАК ЧИТАТЬ СИГНАЛЫ
- «чат ×N» — во сколько раз сообщений больше обычного; реакции: funny — смех, hype — круто, shock — шок, fail — провал, cringe — кринж.
- Чат отстаёт от события на 5–10 секунд. «громкость +X дБ» — стример громкий. «клипы зрителей» — зрители сами нарезали.
- «тема» — подсказка из предварительного просмотра всего стрима; проверь её по транскрипту.
- Транскрипт автоматический, возможны ошибки распознавания — понимай по смыслу. Мат допустим, его заглушат.

МОНТАЖ ДЛЯ УДЕРЖАНИЯ — думай как монтажёр TikTok, а не как стенографист
Зритель листает ленту и решает за 1–2 секунды, остаться или свайпнуть, а потом каждые несколько секунд решает снова.
Сначала про себя найди в транскрипте:
  1) ЯДРО — ради чего этот клип: панчлайн, поворот, вывод, самая сильная фраза или реакция;
  2) КРЮЧОК — самую раннюю фразу, после которой уже хочется узнать, чем кончится;
  3) МИНИМУМ КОНТЕКСТА — что без чего непонятно (кто, что случилось), и ничего сверх этого;
  4) РАЗВЯЗКУ — где ядро досказано, плюс 1–2 с реакции.
Клип = крючок → минимум контекста → развитие → ядро/развязка → короткая реакция. Всё остальное — в cuts или за границы.

Типичные сценарии:
- Долгая подводка («короче, щас расскажу», «было это давно», «чат, слушайте», повтор вопроса из чата, вступление к донату): начинай сразу с сути. Если для понимания нужна одна фраза из подводки — оставь её, остальное вырежи через cuts.
- История: завязка нужна, но без долгих отступлений. Убирай повторы («ну и вот, короче, я говорю…» два раза), уточнения не по делу, отвлечения на чат/игру посреди рассказа. Концовку НЕ трогай.
- Реакция на видео/новость на экране: начни за 2–5 с до того, что вызвало реакцию (зритель должен увидеть повод), закончи после реакции и короткого вывода стримера, не жди, пока он начнёт следующую тему.
- Мнение/горячий тейк: начинай с самого тезиса или вопроса, на который он отвечает; аргументы — только самые сильные; заканчивай на самой резкой формулировке.
- Объяснение/факт: вопрос → ответ → «вау»-деталь. Без вступлений и повторного объяснения тем же самым.
- Шутка/абсурд: 10–30 с. Всё, что не работает на панчлайн, — лишнее; паузу перед панчлайном не трогай.
- Поиски чего-то, «щас найду», загрузка, тишина, чтение доната не по теме посреди момента — всегда в cuts.
- Если ядро не досказано (стример отвлёкся навсегда, обрыв стрима, тема брошена) — это НЕ клип: score ≤ 40, keep=false.

Длина — следствие, а не цель: шутка 10–30 с, реакция 15–45 с, мнение 20–60 с, история или новость 30–75 с, больше 90 с — только если КАЖДЫЕ несколько секунд появляется что-то новое (автор часто пишет «на минуте уже можно было остановиться», «не на 100 с, а 30–40 с самых интересных моментов»). После вырезок клип должен быть настолько коротким, насколько возможно без потери смысла, но не короче. Как только прозвучала развязка и короткая реакция — конец, без «послесловия» и повторного пересказа.
Проверка: пройди мысленно по клипу и спроси на каждых ~5 секундах «будет ли тут свайп?». Если да — вырежи кусок или сдвинь начало.

ГРАНИЦЫ (start, end — секунды ОТ НАЧАЛА ОКНА кандидата, по таймкодам [сек] транскрипта)
- start — с крючка (или с минимально нужного контекста перед ним). Не «ну короче», «так, чат», не тишина.
- end — после развязки/вывода и короткой реакции; не обрывай на полуслове. ГЛАВНАЯ ОШИБКА, из-за которой автор отклоняет клипы, — «мысль не доведена до конца, клип заканчивается на самом интересном». Дочитай транскрипт: если история или мысль продолжается в блоке «Продолжение после окна» — ставь end там (end может быть больше длины окна). Если концовки нет нигде — score ≤ 40, keep=false.
- start может быть и раньше окна (отрицательный, из блока «Перед окном»), если без этого непонятно, о чём речь.
- Ответ на донат/вопрос чата/сообщение: start — там, где донат приходит или зачитывается (голос озвучки доната, «… пишет», «донат от …», вопрос зрителя), а НЕ на ответе стримера. Иначе непонятно, на что он отвечает: автор отклонял такие клипы как «начало обрезано, не хватило контекста».
- Строка «(тишина N с — … упал стрим)» — обрыв. Клип не должен проходить через обрыв: закончи до него или не бери момент.
- Паузы внутри речи потом вырежутся автоматически, их не нужно учитывать.

ВЫРЕЗКИ ВНУТРИ КЛИПА (cuts)
- cuts — список отрезков [от, до] (секунды от начала окна, как start/end), которые нужно ВЫРЕЗАТЬ: затянутая подводка, чтение доната/сообщения не по теме, «так, чат, подождите», поиск в браузере или игре без комментариев по делу, повтор той же мысли второй раз, отвлечения на постороннее, долгое «эээ… короче…», уточнения, без которых смысл не теряется.
- Режь только целыми фразами (по таймкодам строк транскрипта), чтобы склейка была незаметна и речь после склейки звучала связно. Не режь крючок, ядро, развязку, реакцию и то, без чего непонятно.
- Не бойся резать: обычно 0–6 вырезок; каждая не короче ~1.5 с. Если вырезать нечего — [].

ВИЗУАЛ (visual) — что показать в кадре; лицо стримера видно всегда:
- "talk" — стример рассказывает, рассуждает, отвечает чату; что на экране, для смысла не важно → весь клип только вебка;
- "screen" — смысл в том, что на экране: реакция на видео/стрим/новость, игра, переписка, статья, картинка → вебка + экран;
- "mixed" — экран нужен только местами (например, сначала показывает ролик, потом долго обсуждает): screen_spans — отрезки [от, до] (секунды от начала окна), когда экран нужен; в остальное время — только вебка.
Если сомневаешься — "mixed": первые секунды с экраном, пока зритель понимает, о чём речь.

ЧТО АВТОР УЖЕ ОТКЛОНЯЛ (обобщённо)
- клип кончается до развязки или стрим оборвался, а продолжения нет;
- затянуто: стример долго ищет, повторяется, ходит вокруг да около;
- описательный рассказ без поворота, панчлайна, конфликта или полезного вывода («как выглядят косплееры», «как обходил блокировку») — скучно, score ≤ 50;
- непонятно без контекста стрима;
- опасные темы (оружие, ДТП с пострадавшими).

{title_rules}

ХЕШТЕГИ
- hashtags — 3–5 тегов без «#», без пробелов: имя стримера (как его ищут), тема или игра, 1–2 общих (twitch, стрим). На языке стрима.

НЕЛЬЗЯ ВЫКЛАДЫВАТЬ (score ≤ 30, keep=false): реальное насилие и жестокие аварии, оружие, смерть, наркотики, 18+, травля конкретных людей. TikTok такое режет или банит аккаунт, автор такие клипы отклоняет.

ОЦЕНКА score 0–100: 85+ — сильный кандидат в рекомендации; 70–84 — хороший; 55–69 — средний; <55 — не брать. Будь строг и честен: обычно сильны лишь 20–40% кандидатов. keep = score ≥ 55.
reason — одно предложение: почему зайдёт (или нет) в TikTok.
{streamer_context}"""

TOOL = {
    "name": "submit_clips",
    "description": "Вернуть оценку и параметры для каждого кандидата",
    "input_schema": {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "keep": {"type": "boolean"},
                        "score": {"type": "integer", "minimum": 0, "maximum": 100},
                        "category": {"type": "string", "enum": CATEGORIES},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "cuts": {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                                                           "minItems": 2, "maxItems": 2}},
                        "visual": {"type": "string", "enum": ["talk", "screen", "mixed"]},
                        "screen_spans": {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                                                                   "minItems": 2, "maxItems": 2}},
                        "titles": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "keep", "score", "category", "start", "end", "titles", "hashtags", "reason"],
                },
            }
        },
        "required": ["clips"],
    },
}

JSON_INSTRUCTION = """

ФОРМАТ ОТВЕТА: строго один JSON-объект без пояснений и без markdown:
{"clips":[{"id":"<номер кандидата ровно как в заголовке «### Кандидат …»>","keep":true,"score":72,"category":"story","start":31.5,"end":78.0,"cuts":[[44.0,51.5]],"visual":"mixed","screen_spans":[[31.5,40.0]],"titles":["...","..."],"hashtags":["..."],"reason":"..."}]}
category — одно из: """ + ", ".join(CATEGORIES) + ". Верни запись для КАЖДОГО кандидата."


# ------------------------------------------------------------------ prompt
def transcript_lines(words: list[dict], start: float, max_chars: int, peak: float) -> list[str]:
    """Строки «[сек] текст»: новая строка на паузе > 0.6 с, конце предложения или каждые ~12 слов."""
    lines: list[tuple[float, str]] = []
    cur: list[str] = []
    cur_t = None
    last_e = None
    for w in words:
        if cur and (w["s"] - last_e > 0.6 or len(cur) >= 16 or (cur[-1].endswith((".", "!", "?")) and len(cur) >= 3)):
            lines.append((cur_t, " ".join(cur)))
            cur = []
        if last_e is not None and w["s"] - last_e > 25:
            lines.append((last_e, f"(тишина {w['s'] - last_e:.0f} с — пропал звук, перерыв или упал стрим)"))
        if not cur:
            cur_t = w["s"]
        cur.append(w["w"])
        last_e = w["e"]
    if cur:
        lines.append((cur_t, " ".join(cur)))
    # обрезка: оставляем строки ближе к пику
    while sum(len(t) + 8 for _, t in lines) > max_chars and len(lines) > 3:
        if abs(lines[0][0] - peak) > abs(lines[-1][0] - peak):
            lines.pop(0)
        else:
            lines.pop()
    return [f"[{t - start:.1f}] {txt}" for t, txt in lines]


BEFORE_SEC, AFTER_SEC = 40.0, 90.0   # сколько транскрипта показываем до и после окна кандидата


def _plain_lines(words: list[dict], origin: float) -> list[str]:
    """Строки «[сек] текст» без обрезки (для контекста до/после окна), с пометкой длинной тишины."""
    out, cur, cur_t, last_e = [], [], None, None
    for w in words:
        if cur and (w["s"] - last_e > 0.6 or len(cur) >= 14 or cur[-1].endswith((".", "!", "?"))):
            out.append(f"[{cur_t - origin:.1f}] " + " ".join(cur))
            cur = []
        if last_e is not None and w["s"] - last_e > 25:
            out.append(f"[{last_e - origin:.1f}] (тишина {w['s'] - last_e:.0f} с — пропал звук, перерыв или упал стрим)")
        if not cur:
            cur_t = w["s"]
        cur.append(w["w"])
        last_e = w["e"]
    if cur:
        out.append(f"[{cur_t - origin:.1f}] " + " ".join(cur))
    return out


def candidate_block(c: dict, words: list[dict], cfg: dict) -> str:
    st = c["start"]
    w = [x for x in slice_words(words, c["start"], c["end"]) if x["s"] >= st - 0.05]
    sig = []
    if c.get("topic"):
        sig.append(f"тема: {c['topic']}" + (f" ({c['kind']})" if c.get("kind") else ""))
    if "chat_ratio" in c:
        sig.append(f"чат ×{c['chat_ratio']}")
    if c.get("reaction"):
        sig.append("реакции: " + ", ".join(f"{k} {int(v*100)}%" for k, v in list(c["reaction"].items())[:3]))
    if c.get("loud_delta_db") is not None:
        sig.append(f"громкость +{c['loud_delta_db']} дБ")
    if c.get("viewer_clips"):
        sig.append(f"клипы зрителей: {c['viewer_clips']}")
    out = [f"### Кандидат {c['id']}  (время в стриме {fmt_time(st)}–{fmt_time(c['end'])}, окно {c['end'] - st:.0f} с)",
           f"Подсказки: {'; '.join(sig) or 'нет'}."]
    lines = transcript_lines(w, st, cfg["llm"]["max_transcript_chars"], c["peak"])
    before = [x for x in slice_words(words, st - BEFORE_SEC, st) if x["s"] < st - 0.05]
    after = [x for x in slice_words(words, c["end"], c["end"] + AFTER_SEC) if x["s"] >= c["end"]]
    if before:
        out.append("Перед окном (только для понимания контекста):")
        out.extend(_plain_lines(before, st)[-6:])
    out.append("Транскрипт:")
    out.extend(lines or ["(речи нет)"])
    if after:
        out.append("Продолжение после окна (если мысль не закончилась — продли end сюда):")
        out.extend(_plain_lines(after, st)[:int(cfg["llm"].get("max_after_lines", 10))])
    if c.get("chat"):
        out.append("Чат (время от начала окна, отстаёт от событий):")
        seen = set()
        n_chat = int(cfg["llm"].get("max_chat_lines", 10))
        for t, user, text in c["chat"]:
            key = str(text).strip().lower()[:40]
            if key in seen:  # одинаковые смайлы/пасты подряд ничего не добавляют
                continue
            seen.add(key)
            out.append(f"[{t - st:.0f}] {str(text)[:120]}")
            if len(seen) >= n_chat:
                break
    return "\n".join(out)


def build_prompt(meta: dict, cands: list[dict], words: list[dict], cfg: dict) -> tuple[str, str]:
    cl = cfg["clips"]
    from .scan import short_context
    # описание стримера без markdown-разметки; длинное — до разумного предела (имена и мемы — в начале)
    ctx = short_context(cfg["llm"].get("streamer_context") or "", int(cfg["llm"].get("context_chars", 2500)))
    system = SYSTEM_PROMPT.format(
        today=time.strftime("%d.%m.%Y"),
        min_d=cl["min_duration"], max_d=cl["max_duration"], max_long=cl["max_duration"],
        title_rules=TITLE_RULES,
        streamer_context=("\nО СТРИМЕРЕ (учитывай мемы, прозвища и контекст):\n" + ctx.strip()) if ctx.strip() else "")
    from .feedback import prompt_block
    fb = prompt_block(cfg["llm"].get("streamer_login") or None)
    if fb:
        system += "\n\n" + fb
    head = f"Стрим: «{meta.get('title','')}», канал: {meta.get('channel','')}. Кандидатов: {len(cands)}.\n\n"
    body = "\n\n".join(candidate_block(c, words, cfg) for c in cands)
    return system, head + body


# ---------------------------------------------------------------- parsing
def extract_json(text: str) -> dict:
    """Достаёт первый JSON-объект из ответа, даже если до/после него есть текст или второй объект."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    dec = json.JSONDecoder()
    best = None
    i = text.find("{")
    while i >= 0:
        try:
            obj, _end = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                if any(k in obj for k in ("clips", "segments")):
                    return obj
                best = best or obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    if best is not None:
        return best
    raise TwitchCutError("В ответе Claude не найден корректный JSON")


def _clean_title(t: str) -> str:
    t = re.sub(r"\s+", " ", str(t or "")).strip().strip('"«»').strip()
    t = re.sub(r"\s*#\S+", "", t).strip()
    return t[:120]


def _norm_cuts(raw) -> list[list[float]]:
    out = []
    for x in raw or []:
        try:
            a, b = float(x[0]), float(x[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if b - a >= 0.8:
            out.append([round(a, 2), round(b, 2)])
    return sorted(out)


def _match_ids(rows: list[dict], cands: list[dict]) -> list[tuple[str, dict]]:
    """Сопоставляет ответы Claude кандидатам. Claude иногда теряет префикс номера («c07» вместо «L1c07»)
    или нумерует по порядку («c01, c02…» вместо «L3c13…»): такие ответы раньше выбрасывались, и
    оплаченная оценка заменялась эвристикой. Теперь: точное совпадение → по окончанию → по порядку."""
    ids = [c["id"] for c in cands]
    out, used = [], set()
    pending = []
    for r in rows:
        cid = str(r.get("id", "")).strip()
        if cid in ids and cid not in used:
            out.append((cid, r))
            used.add(cid)
        else:
            pending.append((cid, r))
    rest = []
    for cid, r in pending:
        suf = [i for i in ids if i not in used and cid and i.lower().endswith(cid.lower())]
        if len(suf) == 1:
            out.append((suf[0], r))
            used.add(suf[0])
        else:
            rest.append(r)
    free = [i for i in ids if i not in used]
    if rest and len(rest) == len(free):  # остальные — строго по порядку
        out += list(zip(free, rest))
    elif rest:
        log.warning("Claude вернул %d оценок с непонятными номерами — пропускаю их", len(rest))
    return out


def normalize_result(data: dict, cands: list[dict]) -> dict[str, dict]:
    by_id = {c["id"]: c for c in cands}
    out = {}
    for cid, r in _match_ids(list(data.get("clips", []) or []), cands):
        if cid not in by_id:
            continue
        try:
            titles = [_clean_title(t) for t in (r.get("titles") or []) if _clean_title(t)]
            if not titles and r.get("title"):
                titles = [_clean_title(r["title"])]
            out[cid] = {
                "keep": bool(r.get("keep", True)),
                "score": int(max(0, min(100, float(r.get("score", 0))))),
                "category": r.get("category") if r.get("category") in CATEGORIES else "other",
                "rel_start": float(r.get("start", 0)),
                "rel_end": float(r.get("end", 0)),
                "rel_cuts": _norm_cuts(r.get("cuts")),
                "visual": r.get("visual") if r.get("visual") in ("talk", "screen", "mixed") else None,
                "rel_screen": _norm_cuts(r.get("screen_spans")),
                "titles": titles[:2],
                "title": titles[0] if titles else "",
                "hook": "",
                "description": str(r.get("description") or "").strip(),
                "hashtags": [str(h).lstrip("#").strip().replace(" ", "") for h in (r.get("hashtags") or [])
                             if str(h).strip()],
                "reason": str(r.get("reason") or "").strip(),
            }
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------- backends
def _price_key(model: str) -> str:
    m = model.lower()
    return "opus" if "opus" in m else "sonnet" if "sonnet" in m else "haiku"


def call_api(system: str, user: str, cfg: dict, usage: dict, tool: dict | None = None,
             model: str | None = None) -> dict:
    key = (cfg["llm"].get("api_key") or os.environ.get(cfg["llm"]["api_key_env"], "")).strip()
    if not key:
        raise TwitchCutError(
            f"Не задан ключ API: укажите llm.api_key в config.yaml или переменную окружения {cfg['llm']['api_key_env']}. "
            "Получите ключ на console.anthropic.com или переключите llm.backend на claude-cli / manual / none.")
    model = model or cfg["llm"]["model"]
    tool = tool or TOOL
    body = {
        "model": model, "max_tokens": 8000, "system": system,
        "messages": [{"role": "user", "content": user}],
        "tools": [tool], "tool_choice": {"type": "tool", "name": tool["name"]},
    }
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    for attempt in range(6):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=300)
        except requests.RequestException as e:
            log.warning("API: сетевая ошибка %s, повтор", e)
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            _add_usage(usage, data.get("usage") or {})
            for block in data.get("content", []):
                if block.get("type") == "tool_use":
                    return block.get("input") or {}
                if block.get("type") == "text" and "{" in block.get("text", ""):
                    return extract_json(block["text"])
            raise TwitchCutError("API вернул ответ без данных")
        if r.status_code in (429, 500, 502, 503, 529):
            wait = float(r.headers.get("retry-after") or 5 * (attempt + 1))
            log.warning("API занят (%s), жду %.0f с", r.status_code, wait)
            time.sleep(wait)
            continue
        raise TwitchCutError(f"Ошибка Anthropic API {r.status_code}: {r.text[:500]}")
    raise TwitchCutError("Anthropic API недоступен после нескольких попыток")


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "requests")


def _add_usage(usage: dict, u: dict) -> None:
    """Учёт токенов. input_tokens — всё, что Claude прочитал (как и раньше); cache_read_tokens — сколько из
    этого пришло из кэша (повтор того же системного промпта в течение нескольких минут: в API такой вход
    стоит в 10 раз дешевле обычного)."""
    cr = int(u.get("cache_read_input_tokens") or 0)
    usage["input_tokens"] = usage.get("input_tokens", 0) + int(u.get("input_tokens") or 0) + cr \
        + int(u.get("cache_creation_input_tokens") or 0)
    usage["cache_read_tokens"] = usage.get("cache_read_tokens", 0) + cr
    usage["output_tokens"] = usage.get("output_tokens", 0) + int(u.get("output_tokens") or 0)
    usage["requests"] = usage.get("requests", 0) + 1


def find_claude() -> str | None:
    """Ищет Claude Code: PATH, нативный установщик, winget, npm."""
    exe = shutil.which("claude")
    if exe:
        return exe
    if os.name == "nt":
        home = Path(os.environ.get("USERPROFILE", ""))
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        appdata = Path(os.environ.get("APPDATA", ""))
        for c in [home / ".local" / "bin" / "claude.exe", local / "Microsoft" / "WinGet" / "Links" / "claude.exe",
                  appdata / "npm" / "claude.cmd", local / "Programs" / "claude" / "claude.exe"]:
            if c.is_file():
                return str(c)
        for c in (local / "Microsoft" / "WinGet" / "Packages").glob("Anthropic.ClaudeCode*/**/claude.exe"):
            return str(c)
    else:
        c = Path.home() / ".local" / "bin" / "claude"
        if c.is_file():
            return str(c)
    return None


_status_cache: dict = {}


def claude_status(force: bool = False) -> dict:
    """Установлен ли Claude Code и выполнен ли вход (для страницы «Статус»)."""
    if not force and _status_cache.get("t", 0) > time.time() - 60:
        return _status_cache["v"]
    exe = find_claude()
    st = {"installed": bool(exe), "logged_in": False, "version": "", "method": ""}
    if exe:
        kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
        try:
            v = subprocess.run([exe, "--version"], capture_output=True, timeout=30, **kwargs)
            st["version"] = v.stdout.decode("utf-8", "replace").strip()
            a = subprocess.run([exe, "auth", "status"], capture_output=True, timeout=30, **kwargs)
            st["logged_in"] = a.returncode == 0
            try:
                st["method"] = json.loads(a.stdout.decode("utf-8", "replace")).get("authMethod", "")
            except ValueError:
                pass
        except (OSError, subprocess.TimeoutExpired) as e:
            st["error"] = str(e)
    _status_cache.update(t=time.time(), v=st)
    return st


def call_claude_cli(system: str, user: str, cfg: dict, usage: dict, workdir: Path | None = None,
                    json_instruction: str | None = None, model: str | None = None) -> dict:
    """Запрос через Claude Code в режиме `-p` — расходует лимиты подписки Pro/Max, а не деньги за API."""
    exe = find_claude()
    if not exe:
        raise TwitchCutError("Claude Code не установлен. Запустите setup.bat ещё раз — он установит его и попросит войти.")
    workdir = workdir or Path.cwd()
    sys_file = workdir / f"llm_system_prompt_{os.getpid()}_{threading.get_ident()}.txt"
    sys_file.write_text(system, encoding="utf-8")
    model = model or cfg["llm"].get("cli_model") or "sonnet"
    prompt = user + (json_instruction if json_instruction is not None else JSON_INSTRUCTION)
    full = [exe, "-p", "--output-format", "json", "--model", model,
            "--system-prompt-file", str(sys_file), "--tools", "", "--no-session-persistence"]
    basic = [exe, "-p", "--output-format", "json", "--model", model]
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    p = None
    for attempt, (cmd, inp) in enumerate([(full, prompt), (basic, system + "\n\n" + prompt)]):
        try:
            p = subprocess.run(cmd, input=inp.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=1200, cwd=str(workdir), **kwargs)
        except subprocess.TimeoutExpired as e:
            raise TwitchCutError("Claude Code не ответил за 20 минут") from e
        err = (p.stderr or b"").decode("utf-8", "replace") + (p.stdout or b"").decode("utf-8", "replace")[:2000]
        if p.returncode == 0:
            break
        low = err.lower()
        if any(k in low for k in ("login", "log in", "not logged", "authenticat", "/login", "invalid api key")):
            raise TwitchCutError("Claude Code не авторизован. Запустите login_claude.bat и войдите своей подпиской.")
        if any(k in low for k in ("usage limit", "rate limit", "limit reached", "5-hour")):
            raise TwitchCutError("Достигнут лимит подписки Claude. Подождите сброса лимита и нажмите «Переанализировать».")
        if attempt == 0 and ("unknown option" in low or "error: option" in low or "unexpected argument" in low):
            log.info("Старая версия Claude Code — повторяю с базовыми параметрами")
            continue
        raise TwitchCutError("Claude Code завершился с ошибкой: " + err[-800:])
    try:
        sys_file.unlink()
    except OSError:
        pass
    raw = p.stdout.decode("utf-8", "replace")
    try:
        env = json.loads(raw)
        if env.get("is_error"):
            raise TwitchCutError("Claude Code: " + str(env.get("result"))[:500])
        text = env.get("result") or ""
        _add_usage(usage, env.get("usage") or {})
        usage["model"] = model
    except json.JSONDecodeError:
        text = raw
    return extract_json(text)


def call_claude_cli_blocks(system: str, blocks: list[dict], cfg: dict, usage: dict, workdir: Path | None = None,
                           model: str | None = None, timeout: int = 600) -> dict:
    """Запрос с картинками через Claude Code (подписка): сообщение в формате API (текст + изображения base64)
    передаётся через --input-format stream-json."""
    exe = find_claude()
    if not exe:
        raise TwitchCutError("Claude Code не установлен")
    workdir = workdir or Path.cwd()
    sys_file = workdir / f"llm_vision_prompt_{os.getpid()}_{threading.get_ident()}.txt"
    sys_file.write_text(system, encoding="utf-8")
    model = model or cfg["llm"].get("cli_model") or "sonnet"
    cmd = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", model, "--system-prompt-file", str(sys_file), "--tools", "", "--no-session-persistence"]
    msg = {"type": "user", "message": {"role": "user", "content": blocks}}
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    try:
        p = subprocess.run(cmd, input=(json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, cwd=str(workdir), **kwargs)
    except subprocess.TimeoutExpired as e:
        raise TwitchCutError("Claude Code не ответил вовремя") from e
    finally:
        try:
            sys_file.unlink()
        except OSError:
            pass
    out = (p.stdout or b"").decode("utf-8", "replace")
    result = None
    for line in out.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "result":
            result = d
    if result is None:
        err = (p.stderr or b"").decode("utf-8", "replace")[-600:] + out[-400:]
        low = err.lower()
        if any(k in low for k in ("login", "not logged", "authenticat", "invalid api key")):
            raise TwitchCutError("Claude Code не авторизован. Запустите login_claude.bat и войдите своей подпиской.")
        if any(k in low for k in ("usage limit", "rate limit", "limit reached")):
            raise TwitchCutError("Достигнут лимит подписки Claude.")
        raise TwitchCutError("Claude Code (картинки) завершился с ошибкой: " + err)
    if result.get("is_error"):
        raise TwitchCutError("Claude Code: " + str(result.get("result"))[:400])
    _add_usage(usage, result.get("usage") or {})
    return extract_json(result.get("result") or "")


def llm_vision(system: str, blocks: list[dict], cfg: dict, usage: dict, workdir: Path, tool: dict,
               json_instruction: str, cli_model: str | None = None) -> dict:
    """Запрос к Claude с картинками (api или подписка через Claude Code)."""
    blocks = list(blocks)
    if cfg["llm"]["backend"] == "api":
        return call_api(system, blocks, cfg, usage, tool=tool)
    blocks.append({"type": "text", "text": json_instruction})
    return call_claude_cli_blocks(system, blocks, cfg, usage, workdir, model=cli_model)


def llm_call(system: str, user: str, cfg: dict, usage: dict, workdir: Path, tool: dict | None = None,
             json_instruction: str | None = None, cli_model: str | None = None) -> dict:
    """Единая точка вызова Claude для любого бэкенда (api / claude-cli)."""
    if cfg["llm"]["backend"] == "api":
        return call_api(system, user, cfg, usage, tool=tool)
    return call_claude_cli(system, user, cfg, usage, workdir, json_instruction=json_instruction, model=cli_model)


def heuristic(cands: list[dict], words: list[dict], cfg: dict) -> dict[str, dict]:
    """Без LLM: оценка по сигналам, клип заканчивается чуть после пика реакции."""
    target = cfg["clips"]["target_duration"]
    out = {}
    for c in cands:
        st = c["start"]
        cat = next(iter(c.get("reaction") or {}), "other")
        cat = {"hype": "epic", "shock": "scare", "funny": "funny", "fail": "fail",
               "cringe": "cringe", "wholesome": "wholesome"}.get(cat, "other")
        if c.get("kind") in CATEGORIES:
            cat = c["kind"]
        end_rel = min(c["end"] - st, c["peak"] - st + 6)
        out[c["id"]] = {
            "keep": True, "score": int(c.get("pre_score") or (35 + 55 * c.get("signal_norm", 0.5))), "category": cat,
            "rel_start": max(0.0, end_rel - target), "rel_end": end_rel,
            "hook": "", "title": c.get("topic") or f"Момент {fmt_time(c['peak'])}",
            "titles": [c["topic"]] if c.get("topic") else [], "description": "",
            "hashtags": ["twitch", "стрим"], "reason": "оценка только по сигналам чата и звука",
            "heuristic": True,
        }
    return out


def rank_candidates(meta: dict, cands: list[dict], words: list[dict], cfg: dict, job_dir: Path,
                    progress: ProgressFn = noop_progress, wait_manual=None) -> tuple[dict, dict]:
    """Возвращает ({id: оценка}, отчёт об использовании)."""
    backend = cfg["llm"]["backend"]
    usage: dict = {"backend": backend}
    if backend == "none" or not cands:
        return heuristic(cands, words, cfg), usage

    if backend == "manual":
        system, user = build_prompt(meta, cands, words, cfg)
        prompt_path = job_dir / "llm_prompt.txt"
        resp_path = job_dir / "llm_response.txt"
        prompt_path.write_text(system + "\n\n---\n\n" + user + JSON_INSTRUCTION, encoding="utf-8")
        if not resp_path.exists():
            if wait_manual is None:
                raise TwitchCutError("Ручной режим: нет ответа LLM")
            wait_manual(prompt_path, resp_path)
        data = extract_json(resp_path.read_text(encoding="utf-8"))
        res = normalize_result(data, cands)
        return _fill_missing(res, cands, words, cfg), usage

    bs = max(1, int(cfg["llm"]["batch_size"]))
    results: dict = {}
    batches = [cands[i:i + bs] for i in range(0, len(cands), bs)]
    name = "Claude " + (cfg["llm"].get("cli_model") or "sonnet").capitalize() if backend == "claude-cli" else "Claude API"
    import hashlib
    from .util import read_json, write_json
    parts = Path(job_dir) / "rank_parts" if job_dir else None
    failed = 0
    for bi, batch in enumerate(batches):
        progress("llm", bi / len(batches), f"{name} анализирует моменты: пакет {bi+1} из {len(batches)}")
        key = hashlib.md5(("|".join(c["id"] + f"{c['start']:.0f}" for c in batch) + backend
                           + str(cfg["llm"].get("cli_model") or cfg["llm"].get("model"))).encode()).hexdigest()[:10]
        cache = parts / f"batch_{bi:02d}_{key}.json" if parts else None
        data = read_json(cache) if cache else None
        if data is None:
            system, user = build_prompt(meta, batch, words, cfg)
            err = None
            for attempt in range(2):  # один повтор: иногда ответ обрывается или приходит не-JSON
                try:
                    data = call_api(system, user, cfg, usage) if backend == "api" else \
                        call_claude_cli(system, user, cfg, usage, job_dir)
                    err = None
                    break
                except TwitchCutError as e:
                    err = e
                    log.warning("Пакет %d: ответ Claude не принят (%s)%s", bi + 1, e, " — повторяю" if attempt == 0 else "")
                    if "лимит" in str(e).lower() or "limit" in str(e).lower():
                        break
            if err is not None:
                # Не теряем работу: этот пакет оценим по сигналам чата/звука, остальные — Claude
                failed += 1
                usage["warning"] = f"{err} Часть моментов ({failed} из {len(batches)} пакетов) оценена только по чату и звуку."
                usage["fallback"] = True
                results.update(heuristic(batch, words, cfg))
                if failed >= 2 and bi == failed - 1:
                    rest = [c for b in batches[bi + 1:] for c in b]
                    results.update(heuristic(rest, words, cfg))
                    break  # Claude недоступен совсем — не тратим время
                continue
            if cache:
                parts.mkdir(parents=True, exist_ok=True)
                write_json(cache, data)
        results.update(normalize_result(data, batch))
    if backend == "api":
        pin, pout = PRICES[_price_key(cfg["llm"]["model"])]
        usage["cost_usd"] = round(usage.get("input_tokens", 0) / 1e6 * pin + usage.get("output_tokens", 0) / 1e6 * pout, 4)
    progress("llm", 1.0, "Анализ завершён")
    return _fill_missing(results, cands, words, cfg), usage


def _fill_missing(res: dict, cands: list[dict], words: list[dict], cfg: dict) -> dict:
    missing = [c for c in cands if c["id"] not in res]
    if missing:
        log.warning("LLM не вернул оценку для %d кандидатов — использую эвристику", len(missing))
        h = heuristic(missing, words, cfg)
        for k, v in h.items():
            v["score"] = min(v["score"], 45)
            res[k] = v
    return res


# ------------------------------------------------------------- «ужать» клип
SHORTEN_SYSTEM = """Ты — монтажёр коротких роликов для TikTok. Тебе дан транскрипт уже смонтированного клипа со стрима
(таймкоды [сек] — от начала клипа). Задача — УЖАТЬ клип примерно до {target} секунд итоговой длины, оставив только
самое сильное: крючок в начале, минимально нужный контекст, ядро (панчлайн, поворот, вывод) и короткую реакцию.

Как ужимать:
- сдвинь start на самую цепляющую фразу, если начало можно понять без предыдущего;
- сдвинь end сразу после развязки и короткой реакции;
- внутри вырежи (cuts) всё, без чего смысл сохраняется: подводки, повторы, уточнения, отвлечения, поиски, чтение чата не по теме;
- режь только целыми фразами по таймкодам строк, чтобы склейки звучали связно;
- НЕ вырезай развязку/панчлайн и то, без чего непонятно, о чём речь.
Итоговая длина = (end − start) − сумма длин cuts. Она должна быть близка к {target} с (допустимо ±20%).
Если без потери смысла короче не получается — верни минимально возможную длину и объясни в note.
{extra}"""

SHORTEN_TOOL = {
    "name": "submit_edit",
    "description": "Новые границы и вырезки клипа",
    "input_schema": {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "cuts": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
            "note": {"type": "string"},
        },
        "required": ["start", "end", "cuts"],
    },
}

SHORTEN_JSON = """

ФОРМАТ ОТВЕТА: строго один JSON-объект без пояснений и без markdown:
{"start":3.2,"end":41.0,"cuts":[[12.5,19.0]],"note":"..."}"""


def shorten_plan(clip: dict, words: list[dict], target: float, cfg: dict, workdir: Path,
                 usage: dict | None = None) -> dict:
    """Просит Claude ужать клип до ~target секунд. Возвращает {start, end, cuts, note} в секундах от начала клипа."""
    st, en = float(clip["start"]), float(clip["end"])
    ws = [w for w in words if st - 0.05 <= (w["s"] + w["e"]) / 2 <= en]
    if len(ws) < 5:
        raise TwitchCutError("У клипа слишком мало речи, чтобы ужимать его по смыслу")
    cur_cuts = [[round(a - st, 1), round(b - st, 1)] for a, b in (clip.get("content_cuts") or [])]
    lines = _plain_lines(ws, st)
    extra = ""
    if clip.get("title"):
        extra += f"\nО чём клип: «{clip['title']}»."
    if clip.get("reason"):
        extra += f"\nПочему его выбрали: {clip['reason']}"
    system = SHORTEN_SYSTEM.format(target=int(target), extra=extra)
    user = (f"Текущая длина клипа: {en - st:.0f} с (после вырезания пауз ~{clip.get('final_duration') or en - st:.0f} с). "
            f"Уже вырезано: {cur_cuts or 'ничего'}.\nНужно: ~{int(target)} с.\n\nТранскрипт клипа:\n" + "\n".join(lines))
    data = llm_call(system, user, cfg, usage if usage is not None else {}, workdir, tool=SHORTEN_TOOL,
                    json_instruction=SHORTEN_JSON)
    if "start" not in data and isinstance(data.get("clips"), list) and data["clips"]:
        data = data["clips"][0]
    try:
        s, e = float(data["start"]), float(data["end"])
    except (KeyError, TypeError, ValueError) as ex:
        raise TwitchCutError("Claude вернул непонятный ответ на «ужать»") from ex
    dur = en - st
    s, e = max(0.0, min(s, dur)), max(0.0, min(e, dur))
    if e - s < 3:
        raise TwitchCutError("Claude предложил слишком короткий клип")
    cuts = [[max(s, a), min(e, b)] for a, b in _norm_cuts(data.get("cuts")) if min(e, b) - max(s, a) >= 0.8]
    return {"start": s, "end": e, "cuts": cuts, "note": str(data.get("note") or "").strip()}
