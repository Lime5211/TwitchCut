"""ИИ-монтажёр: Claude СМОТРИТ клип целиком (кадры каждые ~2 с + расшифровка с паузами) и решает всё сразу,
как человек за монтажным столом:
  - где начать (чтобы с первой секунды было понятно, о чём речь: вопрос, донат, то, что показали на экране),
  - где закончить (сразу после развязки и реакции),
  - что вырезать (вода, повторы, поиски) и что НЕЛЬЗЯ резать, хотя там тишина (на экране происходит смешное),
  - что показывать в каждый момент (только вебка / вебка + экран / только экран) и когда менять кадр.
Второй короткий запрос — «кадрировщик»: на крупных кадрах с сеткой Claude обводит окно видео/игры, о котором
идёт речь, а программа подтягивает рамку к настоящим краям окна. Экран потом показывается от верхней до нижней
границы этого окна, по его центру — без «съезжания» влево и без полосок сверху.

Время везде — секунды от начала клипа, который предложил оценщик (может быть отрицательным: начало раньше).
Расход: ~10–15 тыс. токенов на клип (6–7 картинок + текст) и ~3 тыс. на кадрирование.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .util import log, read_json, write_json

VERSION = 1
TW, TH = 426, 240          # плитка обзорного листа (3×3 на картинке)
BW, BH = 960, 540          # крупный кадр для кадрировщика

EDITOR_SYSTEM = """Ты — опытный монтажёр вертикальных роликов TikTok из Twitch-стримов. Тебе показан фрагмент стрима
целиком: кадры каждые ~2 секунды (подпись — секунды от начала предложенного клипа; минус — раньше), жёлтая рамка —
вебка стримера, зелёная полоска внизу кадра — его сейчас предлагают включить в клип. И расшифровка речи с таймкодами,
где отмечены паузы. Сделай из этого ролик, который досмотрят до конца.

ГРАНИЦЫ
- start — первая секунда, с которой зрителю, НИЧЕГО не знающему о стриме, понятно, что происходит. Если стример
  отвечает на донат/вопрос/сообщение или реагирует на то, что на экране, — start там, где это появилось (донат
  зачитывают или он виден на экране, видео/картинка начались), а НЕ на его ответе. Смотри на кадры: повод часто
  виден только на экране. Автор постоянно жалуется «начало обрезано, непонятно».
- end — сразу после развязки (панчлайн, вывод, чем кончилось) и 1–2 с реакции. Если после первого панчлайна идёт
  продолжение той же истории с новым поворотом — включи его. Не тяни дальше: «послесловие», повторный пересказ,
  переход к другой теме — уже за концом.
- Можно расширять за зелёную зону (кадры и текст до/после показаны), если иначе непонятно или нет концовки.

ДЛИНА — главная претензия автора: «почти всегда хочется ужать в 1.5–2 раза». Итоговая длина = end − start − вырезки.
Шутка/реакция 15–35 с, мнение 20–45 с, история/новость 30–60 с. Больше 70 с — только если КАЖДЫЕ 5 секунд новое.
- cuts — что вырезать внутри: подводки и «короче, щас», повторы той же мысли, поиски («щас найду», листает,
  грузится), отвлечения на чат/донаты не по теме, уточнения без которых смысл не теряется, долгие «эээ».
  Режь целыми фразами по таймкодам. Не трогай крючок, повод (донат/то, что на экране), ядро, развязку и реакцию.
- keep — НЕЛЬЗЯ вырезать, хотя стример молчит: на экране/вебке в эти секунды происходит что-то важное или смешное
  (в игре что-то выскочило/сломалось, в видео панчлайн, стример молча корчится/смеётся, что-то показывает).
  Паузы в речи программа вырезает сама — перечисли такие места, иначе их выкинут. Сверяйся с кадрами в паузах.

КАДР (shots — покрывают от start до end по порядку):
- "split" — вебка сверху + экран снизу: смысл связан с тем, что на экране (реакция на видео/картинку/статью, игра,
  донат или сообщение на экране, товар, переписка). what — что на экране главное: "video", "game", "page" (сайт,
  статья, картинка, магазин), "chat", "donation", "other".
- "cam" — только вебка: стример рассказывает о своём, а экран для смысла не нужен (статичный, посторонний, пустой).
- "camwide" — только вебка общим планом: он показывает что-то в камеру (телефон, предмет) или активно жестикулирует.
- "screen" — только экран, без лица: редко, только когда стример молчит, а смотреть надо на экран.
- Пока смысл в экране — держи "split" ВСЁ это время, а не только первые секунды. Не переключай на "cam", если зритель
  после этого перестанет понимать, о чём речь. В конце клипа не уходи с экрана, если реакция всё ещё на него.
- Меняй кадр только когда меняется то, что важно; кадр не короче 6 с; обычно 1–3 кадра на клип. Не делай «вебка →
  экран → вебка» без явной причины. Если весь клип он рассказывает о своём — один кадр "cam".
- Донат/всплывашка не по теме — не переключай кадр ради неё (лучше вырежи этот кусок).
{notes}
Ответ — только JSON."""

EDITOR_JSON = """

ФОРМАТ ОТВЕТА: строго один JSON-объект без пояснений и markdown:
{"start":<с>,"end":<с>,"cuts":[[<с>,<с>]],"keep":[[<с>,<с>]],"shots":[{"from":<с>,"to":<с>,"layout":"split","what":"video","why":"…"},{"from":<с>,"to":<с>,"layout":"cam","why":"…"}],"note":"одной фразой, о чём клип и почему так смонтирован"}
Вместо <с> — секунды по подписям кадров и таймкодам расшифровки."""

EDITOR_TOOL = {
    "name": "submit_edit",
    "description": "Монтаж клипа",
    "input_schema": {
        "type": "object",
        "properties": {
            "start": {"type": "number"}, "end": {"type": "number"},
            "cuts": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "keep": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "shots": {"type": "array", "items": {"type": "object", "properties": {
                "from": {"type": "number"}, "to": {"type": "number"},
                "layout": {"type": "string", "enum": ["cam", "camwide", "split", "screen"]},
                "what": {"type": "string", "enum": ["video", "game", "page", "chat", "donation", "other"]},
                "why": {"type": "string"}}, "required": ["from", "to", "layout"]}},
            "note": {"type": "string"},
        },
        "required": ["start", "end", "shots"],
    },
}

FRAMER_SYSTEM = """Тебе показаны крупные кадры Twitch-стрима с сеткой (линии через каждые 0.1 кадра, подписи — доли
кадра по горизонтали x и вертикали y). Жёлтая рамка — вебка стримера (её не включай). Голубая рамка «A», если есть, —
то, что нашла программа.
Для каждого кадра обведи то, что нужно показать зрителю (указано под номером кадра): окно видео-плеера, окно игры,
страницу/статью/картинку, колонку чата или плашку доната. Рамка — ТОЧНО по краям этого окна: верхний край окна,
нижний край окна, левый и правый край, без полей браузера, вкладок, панели поиска, рамок и подписей вокруг.
Подписи и текст, которые являются частью видео/картинки (мем-подпись, субтитры ролика), — ВНУТРИ рамки.
Если голубая рамка «A» уже точно обводит нужное — верни "A". Если важное — весь экран (игра во весь экран) — [0,0,1,1].
Меряй по сетке для каждого кадра отдельно."""

FRAMER_JSON = """

ФОРМАТ ОТВЕТА: строго один JSON-объект: {"rects":[{"n":1,"rect":[x,y,w,h]},{"n":2,"rect":"A"}]}
x, y — левый верхний угол, w, h — ширина и высота, в долях кадра (0..1), измеренные по сетке."""

FRAMER_TOOL = {
    "name": "submit_rects",
    "description": "Рамки окон",
    "input_schema": {"type": "object", "properties": {"rects": {"type": "array", "items": {"type": "object", "properties": {
        "n": {"type": "integer"}, "rect": {}}, "required": ["n", "rect"]}}}, "required": ["rects"]},
}


def enabled(cfg: dict) -> bool:
    rc = cfg.get("editor") or {}
    return bool(rc.get("enabled", True)) and cfg["llm"]["backend"] in ("api", "claude-cli")


def cached(path: Path) -> dict | None:
    old = read_json(path) if path else None
    if old and old.get("plan") and int(old.get("v") or 0) == VERSION:
        return old["plan"]
    return None


# ------------------------------------------------------------------ картинки
def _label(im: np.ndarray, text: str, scale: float = 0.6) -> None:
    import cv2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(im, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(im, text, (5, th + 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)


def _sheet_tiles(frames: list, cam_box: list | None, inside: tuple[float, float]) -> list[bytes]:
    """Обзорные листы 3×3 (по 9 кадров) с подписью времени, рамкой вебки и зелёной полоской «в клипе»."""
    import cv2
    tiles = []
    for t, fr in frames:
        im = cv2.resize(fr, (TW, TH), interpolation=cv2.INTER_AREA)
        if cam_box:
            x, y, w, h = cam_box[:4]
            cv2.rectangle(im, (int(x * TW), int(y * TH)), (int((x + w) * TW) - 1, int((y + h) * TH) - 1), (0, 230, 255), 1)
        if inside[0] - 0.5 <= t <= inside[1] + 0.5:
            cv2.rectangle(im, (0, TH - 5), (TW, TH), (0, 200, 0), -1)
        _label(im, f"{t:.0f}s")
        tiles.append(im)
    out = []
    for i in range(0, len(tiles), 9):
        group = tiles[i:i + 9]
        while len(group) < 9:
            group.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(group[r * 3:(r + 1) * 3]) for r in range(3)]
        grid = np.vstack(rows)
        for k in (1, 2):
            cv2.line(grid, (k * TW, 0), (k * TW, 3 * TH), (40, 40, 40), 2)
            cv2.line(grid, (0, k * TH), (3 * TW, k * TH), (40, 40, 40), 2)
        ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append(buf.tobytes())
    return out


def _grid_frame(fr: np.ndarray, cam_box: list | None, cand: list | None, title: str) -> bytes:
    """Крупный кадр с сеткой 0.1 для точной рамки."""
    import cv2
    im = cv2.resize(fr, (BW, BH), interpolation=cv2.INTER_AREA)
    over = im.copy()
    for k in range(1, 10):
        x, y = int(BW * k / 10), int(BH * k / 10)
        cv2.line(over, (x, 0), (x, BH), (255, 255, 255), 1)
        cv2.line(over, (0, y), (BW, y), (255, 255, 255), 1)
    im = cv2.addWeighted(over, 0.35, im, 0.65, 0)
    for k in range(1, 10):
        x, y = int(BW * k / 10), int(BH * k / 10)
        for pos in ((x - 12, 14), (x - 12, BH - 4)):
            cv2.putText(im, f".{k}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(im, f".{k}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for pos in ((2, y + 5), (BW - 26, y + 5)):
            cv2.putText(im, f".{k}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(im, f".{k}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if cam_box:
        x, y, w, h = cam_box[:4]
        cv2.rectangle(im, (int(x * BW), int(y * BH)), (int((x + w) * BW) - 1, int((y + h) * BH) - 1), (0, 230, 255), 2)
    if cand:
        x, y, w, h = cand[:4]
        cv2.rectangle(im, (int(x * BW), int(y * BH)), (int((x + w) * BW) - 1, int((y + h) * BH) - 1), (255, 200, 0), 2)
        cv2.putText(im, "A", (int(x * BW) + 6, int(y * BH) + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    _label(im, title, 0.7)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def _img(jpg: bytes) -> dict:
    import base64
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpg).decode()}}


# ------------------------------------------------------------------ текст
def transcript_with_pauses(words: list[dict], lo: float, hi: float) -> str:
    """Расшифровка «[сек] текст» (секунды от начала клипа) с отметками пауз ≥ 1.5 с — где стример молчит,
    надо смотреть на кадры: там может происходить важное."""
    ws = [w for w in words if lo - 0.5 <= w["s"] <= hi]
    lines, cur, cur_t, last = [], [], None, None
    for w in ws:
        if last is not None and w["s"] - last >= 1.5:
            if cur:
                lines.append(f"[{cur_t:.1f}] " + " ".join(cur))
                cur = []
            lines.append(f"[{last:.1f}–{w['s']:.1f}] (пауза {w['s'] - last:.0f} с — что на кадрах?)")
        elif cur and (len(cur) >= 14 or cur[-1].endswith((".", "!", "?"))):
            lines.append(f"[{cur_t:.1f}] " + " ".join(cur))
            cur = []
        if not cur:
            cur_t = w["s"]
        cur.append(w["w"])
        last = w["e"]
    if cur:
        lines.append(f"[{cur_t:.1f}] " + " ".join(cur))
    if ws and hi - ws[-1]["e"] >= 1.5:
        lines.append(f"[{ws[-1]['e']:.1f}–{hi:.1f}] (пауза до конца фрагмента)")
    return "\n".join(lines) or "(речи нет)"


# ------------------------------------------------------------------ план
def _spans(raw, lo: float, hi: float, min_len: float = 0.6) -> list[list[float]]:
    out = []
    for c in raw or []:
        try:
            a, b = float(c[0]), float(c[1])
        except (TypeError, ValueError, IndexError):
            continue
        a, b = max(lo, a), min(hi, b)
        if b - a >= min_len:
            out.append([round(a, 2), round(b, 2)])
    out.sort()
    merged: list[list[float]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 0.2:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def normalize_edit(data: dict, lo: float, hi: float, dur: float, fixed: bool = False, min_shot: float = 5.0) -> dict:
    """Проверка ответа монтажёра: границы в пределах доступного видео, шоты без дыр, вырезки — не больше 55 %."""
    try:
        s, e = float(data.get("start", 0.0)), float(data.get("end", dur))
    except (TypeError, ValueError):
        s, e = 0.0, dur
    if fixed:
        s, e = 0.0, dur
    s = max(lo, min(s, hi - 5))
    e = min(hi, max(e, s + 5))
    cuts = _spans(data.get("cuts"), s + 0.3, e - 0.3, 0.8)
    if sum(b - a for a, b in cuts) > 0.55 * (e - s):
        # слишком смело — оставляем самые длинные вырезки, пока не наберётся половина
        cuts.sort(key=lambda c: -(c[1] - c[0]))
        acc, kept = 0.0, []
        for a, b in cuts:
            if acc + (b - a) <= 0.5 * (e - s):
                kept.append([a, b])
                acc += b - a
        cuts = sorted(kept)
    keep = _spans(data.get("keep"), s, e, 0.3)
    # вырезка не может съесть то, что велено сохранить
    cuts = [c for c in cuts if not any(min(c[1], k[1]) - max(c[0], k[0]) > 0.3 for k in keep)]
    shots = []
    for sh in data.get("shots") or []:
        try:
            a, b = float(sh.get("from", s)), float(sh.get("to", e))
        except (TypeError, ValueError):
            continue
        lay = sh.get("layout") if sh.get("layout") in ("cam", "camwide", "split", "screen") else None
        if not lay:
            continue
        a, b = max(s, a), min(e, b)
        if b <= a:
            continue
        what = sh.get("what") if lay in ("split", "screen") else None
        shots.append({"t0": a, "t1": b, "layout": lay, "what": what or ("video" if lay in ("split", "screen") else None),
                      "why": str(sh.get("why") or "")[:140]})
    shots.sort(key=lambda x: x["t0"])
    if not shots:
        shots = [{"t0": s, "t1": e, "layout": "cam", "what": None, "why": ""}]
    shots[0]["t0"] = s
    for i in range(len(shots) - 1):
        shots[i]["t1"] = shots[i + 1]["t0"]
    shots[-1]["t1"] = e

    # короткие шоты сливаем с более длинным соседом (длина — по тому, что останется после вырезок)
    def eff(sh):
        return sh["t1"] - sh["t0"] - sum(max(0.0, min(b, sh["t1"]) - max(a, sh["t0"])) for a, b in cuts)
    changed = True
    while changed and len(shots) > 1:
        changed = False
        for i, sh in enumerate(shots):
            if eff(sh) < min_shot:
                if i == 0:
                    j = 1
                elif i == len(shots) - 1:
                    j = i - 1
                else:
                    j = i - 1 if eff(shots[i - 1]) >= eff(shots[i + 1]) else i + 1
                shots[j]["t0"], shots[j]["t1"] = min(shots[j]["t0"], sh["t0"]), max(shots[j]["t1"], sh["t1"])
                shots.pop(i)
                changed = True
                break
    merged = [shots[0]]
    for sh in shots[1:]:
        if sh["layout"] == merged[-1]["layout"] and sh.get("what") == merged[-1].get("what"):
            merged[-1]["t1"] = sh["t1"]
        else:
            merged.append(sh)
    return {"start": round(s, 2), "end": round(e, 2), "cuts": cuts, "keep": keep, "shots": merged,
            "note": str(data.get("note") or "")[:300]}


def _streamer_notes(cfg: dict, notes: str) -> str:
    ctx = cfg["llm"].get("streamer_context") or ""
    taste = ""
    k = ctx.find("ЧТО ЗАХОДИТ")
    if k >= 0:
        taste = ctx[k:k + 1600]
    out = []
    if taste:
        out.append(taste)
    if notes.strip():
        out.append("ЗАМЕЧАНИЯ АВТОРА ПО МОНТАЖУ:\n" + notes.strip()[:1500])
    return ("\n\n" + "\n\n".join(out)) if out else ""


def edit_clip(src: Path, base_off: float, lo: float, hi: float, dur: float, c: dict, words_rel: list[dict],
              li: dict, cfg: dict, workdir: Path, usage: dict, notes: str = "", cache: Path | None = None,
              fixed: bool = False) -> dict | None:
    """Решение монтажёра. base_off — где в файле src начало клипа; [lo, hi] — какой кусок видео доступен
    (секунды от начала клипа); dur — длина клипа, предложенная оценщиком. words_rel — слова (время от начала
    клипа) на всём доступном куске. fixed — границы и вырезки уже заданы автором («ужать»), решить только кадр."""
    from .llm import llm_vision
    from .render import _sample_keyframes
    rc = cfg.get("editor") or {}
    key = hashlib.md5(json.dumps([VERSION, round(lo, 1), round(hi, 1), round(dur, 1), fixed, li.get("cam_box"),
                                  (c.get("title") or "")[:80], rc.get("model")], ensure_ascii=False).encode()).hexdigest()[:12]
    if cache:
        old = read_json(cache)
        if old and old.get("key") == key:
            return old["plan"]
    sw, sh = li["src_w"], li["src_h"]
    frames = _sample_keyframes(src, base_off + lo, hi - lo, sw, sh, out_w=TW)
    frames = [(round(t + lo, 2), f) for t, f in frames]
    max_n = int(rc.get("max_frames", 63))
    if len(frames) > max_n:
        idx = np.linspace(0, len(frames) - 1, max_n).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx))]
    if len(frames) < 3:
        return None
    sheets = _sheet_tiles(frames, li.get("cam_box"), (0.0, dur))
    sugg = ""
    if c.get("content_cuts"):
        cc = [[round(a - c["start"], 1), round(b - c["start"], 1)] for a, b in c["content_cuts"]]
        sugg = f" Оценщик (читал только текст) предложил вырезать: {cc}."
    task = ("Границы и вырезки уже выбрал автор — start=0, end=длина, cuts и keep можно не трогать; реши только кадр (shots)."
            if fixed else
            f"Оценщик (читал только текст) предложил клип 0–{dur:.0f} с.{sugg} Проверь по кадрам и смонтируй как надо.")
    text = (f"Клип: «{c.get('title') or ''}» ({c.get('category') or ''}). {c.get('reason') or ''}\n"
            f"Доступно видео: от {lo:.0f} до {hi:.0f} с. {task}\n"
            f"Кадры по порядку слева направо, сверху вниз, по 9 на картинке.\n\n"
            f"Расшифровка:\n{transcript_with_pauses(words_rel, lo, hi)}")
    system = EDITOR_SYSTEM.format(notes=_streamer_notes(cfg, notes))
    blocks = [_img(j) for j in sheets] + [{"type": "text", "text": text}]
    data = llm_vision(system, blocks, cfg, usage, workdir, EDITOR_TOOL, EDITOR_JSON,
                      cli_model=rc.get("model") or cfg["llm"].get("cli_model"))
    plan = normalize_edit(data, lo, hi, dur, fixed, float(rc.get("min_shot_seconds", 5.0)))
    try:
        frame_windows(plan, src, base_off, li, cfg, workdir, usage)
    except Exception as e:  # рамка окна не обязательна: без неё — экран целиком
        log.warning("Кадрирование экрана не удалось: %s", e)
    log.info("Монтажёр: клип %.1f–%.1f с (было 0–%.1f), вырезок %d (%.0f с), сохранить паузы: %d, кадры: %s",
             plan["start"], plan["end"], dur, len(plan["cuts"]), sum(b - a for a, b in plan["cuts"]), len(plan["keep"]),
             [(round(s_["t0"]), s_["layout"], s_.get("what")) for s_ in plan["shots"]])
    if cache:
        write_json(cache, {"key": key, "plan": plan, "v": VERSION})
    return plan


# ------------------------------------------------------------------ кадрировщик
def frame_windows(plan: dict, src: Path, base_off: float, li: dict, cfg: dict, workdir: Path, usage: dict) -> None:
    """Для каждого кадра «экран» — точная рамка окна (видео/игра/страница/чат/донат) → shot["focus"]."""
    from .llm import llm_vision
    from .render import _sample_keyframes
    from . import screen
    shots = [s for s in plan["shots"] if s["layout"] in ("split", "screen")]
    if not shots:
        return
    sw, sh = li["src_w"], li["src_h"]
    cam = li.get("cam_box")
    a0, a1 = min(s["t0"] for s in shots), max(s["t1"] for s in shots)
    frames = _sample_keyframes(src, base_off + a0, a1 - a0, sw, sh, out_w=BW)
    frames = [(t + a0, f) for t, f in frames]
    if not frames:
        return
    jobs = []
    for s in shots:
        inside = [x for x in frames if s["t0"] <= x[0] <= s["t1"]] or [min(frames, key=lambda x: abs(x[0] - s["t0"]))]
        # программа: окно, где что-то меняется / отличается от фона сцены
        cand = None
        try:
            small = [(t, f) for t, f in inside]
            rects = screen.window_rects(small, cam, screen.load_bg(cfg["llm"].get("streamer_login") or "",
                                                                    (li.get("scene_face") or None)))
            st = screen.screen_states([t for t, _ in small], rects, small, s["t1"], cam)
            if st:
                cand = max(st, key=lambda q: q["t1"] - q["t0"])["rect"]
        except Exception:
            cand = None
        pick = min(inside, key=lambda x: abs(x[0] - (s["t0"] + 0.4 * (s["t1"] - s["t0"]))))
        jobs.append((s, inside, cand, pick))
    rc = cfg.get("editor") or {}
    rects: dict[int, object] = {}
    if cfg["llm"]["backend"] in ("api", "claude-cli") and rc.get("framer", True):
        what_ru = {"video": "окно видео", "game": "окно игры", "page": "страница/картинка/статья", "chat": "колонка чата",
                   "donation": "плашка доната/сообщения", "other": "то, на что реагирует стример"}
        blocks = []
        for n, (s, _, cand, pick) in enumerate(jobs, 1):
            blocks.append(_img(_grid_frame(pick[1], cam, cand, f"#{n}: {what_ru.get(s.get('what') or 'video')}")))
        blocks.append({"type": "text", "text": "Обведи для каждого кадра нужное окно: " + "; ".join(
            f"#{n} — {what_ru.get(s.get('what') or 'video')} ({s.get('why') or ''})" for n, (s, *_r) in enumerate(jobs, 1))})
        try:
            data = llm_vision(FRAMER_SYSTEM, blocks, cfg, usage, workdir, FRAMER_TOOL, FRAMER_JSON,
                              cli_model=rc.get("model") or cfg["llm"].get("cli_model"))
            for r in data.get("rects") or []:
                try:
                    rects[int(r.get("n"))] = r.get("rect")
                except (TypeError, ValueError):
                    continue
        except Exception as e:
            log.warning("Кадрировщик не ответил: %s", e)
    for n, (s, inside, cand, pick) in enumerate(jobs, 1):
        raw = rects.get(n)
        rect = None
        if isinstance(raw, str) and raw.strip().upper().startswith("A") and cand:
            rect = list(cand)
        elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
            try:
                x, y, w, h = [float(v) for v in raw[:4]]
                if w > 0.05 and h > 0.05:
                    x, y = max(0.0, min(x, 0.95)), max(0.0, min(y, 0.95))
                    rect = [x, y, min(w, 1 - x), min(h, 1 - y)]
            except (TypeError, ValueError):
                rect = None
        if rect is None:
            rect = cand or screen._avoid_cam([0.0, 0.0, 1.0, 1.0], cam)
        if not (rect[2] > 0.97 and rect[3] > 0.97):
            try:
                rect = screen.snap_rect(rect, inside if len(inside) >= 3 else frames)
            except Exception:
                pass
        rect = screen._avoid_cam(rect, cam)
        # чуть внутрь: не захватывать линию рамки окна (иначе сверху «полоска»)
        x, y, w, h = rect
        ix, iy = 0.004 * (w < 0.99), 0.006 * (h < 0.99)
        rect = [round(x + ix, 4), round(y + iy, 4), round(max(0.05, w - 2 * ix), 4), round(max(0.05, h - 2 * iy), 4)]
        whole = (s.get("what") in ("chat", "donation", "page"))
        s["focus"] = rect + ([1] if whole else [])
