"""ИИ-режиссёр: Claude СМОТРИТ кадры клипа (и слушает расшифровку) и решает, как монтировать:
когда показывать только вебку, когда вебку + экран и на чём на экране сделать акцент (приблизить чат,
когда стример выбирает зрителя; донат, на который он отвечает; видео, на которое он реагирует; игру
целиком, а не её кусок). Ещё он один раз за стрим проверяет, правильно ли найдена рамка вебки.

Расход: ~6–8 тыс. токенов на клип (до 3 картинок по 2×2 кадра + расшифровка). Не вызывается, когда стример
просто рассказывает, а экран стоит (director.skip_static_talk). Выключается director.enabled: false.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .util import TwitchCutError, ffmpeg_bin, log, read_json, write_json

VERSION = 5
TILE_W, TILE_H = 640, 360

DIRECTOR_SYSTEM = """Ты — режиссёр монтажа вертикальных роликов (TikTok, 9:16) из Twitch-стримов. Тебе показаны кадры клипа
(подпись t=… — секунды от начала клипа; жёлтая рамка — где на экране вебка стримера; деления по краям — доли кадра
0.1…0.9 для координат) и расшифровка речи с таймкодами. Реши, ЧТО показывать в каждый момент.

Варианты кадра (layout):
- "cam" — только вебка стримера крупно. Когда он рассказывает, рассуждает, отвечает, а на экране нет ничего, что нужно
  видеть для понимания (статичная страница, меню, рабочий стол, чёрный/пустой экран, игра «фоном», не о ней речь).
- "camwide" — вебка общим планом: лицо по центру, но видно руки и то, что у него в руках. Когда он активно
  жестикулирует (показывает размер, разводит руками, изображает что-то), держит предмет у камеры или что-то
  показывает руками. Если показывает в камеру телефон/бумагу/предмет С ТЕКСТОМ или мелкими деталями, которые надо
  разглядеть, — лучше "split" с focus ПЛОТНО вокруг этого предмета на вебке и fit "whole": сверху лицо, снизу предмет крупно.
- "split" — вебка сверху + экран снизу. Когда смысл связан с тем, что на экране: он реагирует на видео/стрим/картинку,
  зачитывает донат или сообщение и отвечает на него, выбирает кого-то в чате, показывает переписку/сайт/товар,
  играет и параллельно комментирует происходящее в игре, приходит смешной донат во время игры.
  Обязательно укажи focus — прямоугольник [x, y, w, h] в долях кадра вокруг того, что важно: окно видео-плеера,
  колонка чата, плашка доната с текстом, окно игры, переписка. Эта область будет показана крупно.
  И укажи fit — как её показать (нижняя зона почти квадратная):
    "zoom" (по умолчанию) — приблизить МАКСИМАЛЬНО, края области обрежутся. Для игры и видео: focus — окно
      игры/плеера, а его центр — там, где главное действие (персонаж, лицо в видео, центр событий).
    "whole" — показать область целиком, без обрезки краёв: текст доната/сообщения, колонка чата, переписка,
      или видео, где важное и слева, и справа. focus тогда — ПЛОТНО вокруг текста/плашки, без пустого места,
      иначе текст станет мелким.
  Не включай в focus вебку стримера (она уже сверху) и лишние панели (вкладки браузера, OBS, рамки). Если важна вся
  игра/видео — focus на всё окно игры/плеера, а не его часть.
  Если стример говорит о КОНКРЕТНОЙ вещи на экране (приборная панель машины, товар, ценник, надпись, человек в видео,
  комментарий, картинка) — focus на эту вещь, а не на всё окно: при "zoom" нижняя зона вырезает из широкого видео лишь
  середину, и то, о чём он говорит, может не попасть в кадр. Проверь по кадрам, где эта вещь, и поставь её в центр focus.
- "screen" — только экран (focus), без лица. РЕДКО: лицо стримера в приоритете, используй, только если реакции/речи
  стримера в этот момент нет, а картинка — главное (например, несколько секунд показывает товар/видео молча).

Правила:
- Стример ИГРАЕТ (на экране идёт игра, что-то двигается) — по умолчанию "split" с игрой, даже если он параллельно
  говорит о другом или читает донаты: зрителю интересно видеть и игру, и лицо. "cam" во время игры — только если
  игра стоит (меню, загрузка, пауза) или он надолго отвернулся от неё ради рассказа.
- Не игра (браузер, сайт, рабочий стол, видео на паузе) и смысл не в экране — "cam": лицо стримера в приоритете.
- Стример РАССКАЗЫВАЕТ свою историю или мнение, а на экране идёт что-то постороннее (видео, стрим, шоу, к которому
  рассказ не относится) — "cam", даже если на экране всё двигается. Экран нужен, только если зритель без него не
  поймёт, о чём речь. Сверяйся с расшифровкой: о чём он говорит — о том, что на экране, или о своём?
- Ниже дана подсказка оценщика, который читал всю расшифровку: что по смыслу нужно показывать. Это сильная
  подсказка: отступай от неё, только если кадры явно говорят обратное.
- focus — только по тому, что РЕАЛЬНО видно на кадрах с этим временем. Не выдумывай место доната или окна, которого
  на кадрах нет. Всплывающий донат виден секунды — если на кадрах его нет, focus на основное окно (игра/видео).
- Если вебка стримера наложена поверх игры/видео (она уже показана сверху), focus выбирай так, чтобы её в нём
  не было или было минимум; не включай колонку чата, если речь не о чате.
- Не дёргай кадр: каждый шот не короче 5 с; обычно 1–3 шота на клип. Меняй кадр, только когда меняется то, что важно.
- Если в начале клипа приходит донат/сообщение, на которое стример отвечает, — первые секунды "split" с focus на донат.
- Донаты и всплывашки НЕ по теме момента не показывай (кадр не переключай ради них). Если они мешают — верни их в cuts.
- Если focus меняется (сначала донат, потом видео) — это разные шоты.
- Все участники разговора должны быть в кадре. Если на вебке несколько человек — "cam"/"split" покажут всех
  автоматически. Если стример говорит с кем-то, кого видно на ЭКРАНЕ (созвон, второй стример или гость в окне,
  стрим другого человека), — "split" с focus на этого человека, пока они разговаривают.
- shots покрывают весь клип от 0 до его длины, по порядку, без пропусков.
Ещё проверь жёлтую рамку вебки: если она не совпадает с картинкой вебки (захватывает чат/экран или обрезает вебку) —
верни cam_box с правильной рамкой [x, y, w, h]; если всё верно или вебки нет на кадрах — null.
{extra}"""

DIRECTOR_TOOL = {
    "name": "submit_shots",
    "description": "План кадров клипа",
    "input_schema": {
        "type": "object",
        "properties": {
            "shots": {"type": "array", "items": {"type": "object", "properties": {
                "from": {"type": "number"}, "to": {"type": "number"},
                "layout": {"type": "string", "enum": ["cam", "camwide", "split", "screen"]},
                "focus": {"type": ["array", "null"], "items": {"type": "number"}},
                "fit": {"type": "string", "enum": ["zoom", "whole"]},
                "why": {"type": "string"}}, "required": ["from", "to", "layout"]}},
            "cuts": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "cam_box": {"type": ["array", "null"], "items": {"type": "number"}},
        },
        "required": ["shots"],
    },
}

DIRECTOR_JSON = """

ФОРМАТ ОТВЕТА: строго один JSON-объект без пояснений и без markdown:
{"shots":[{"from":0,"to":9.5,"layout":"split","focus":[0.02,0.12,0.28,0.16],"fit":"whole","why":"донат, на который он отвечает"},{"from":9.5,"to":30.0,"layout":"split","focus":[0.0,0.0,0.62,0.64],"fit":"zoom","why":"играет и комментирует"},{"from":30.0,"to":41.2,"layout":"cam","focus":null,"why":"рассказывает"}],"cuts":[],"cam_box":null}"""

CAMCHECK_SYSTEM = """Ты проверяешь разметку кадров стрима. На каждом кадре жёлтая рамка должна ТОЧНО обводить картинку
веб-камеры стримера (прямоугольник с видео с его камеры): не захватывать чат, экран, плашки вокруг и не обрезать саму
камеру. Деления по краям — доли кадра 0.1…0.9. Если рамка верна — ok=true. Если нет — ok=false и box — правильная
рамка [x, y, w, h] в долях кадра. Если камеры стримера на кадрах нет — ok=false, box=null."""

CAMCHECK_JSON = """

ФОРМАТ ОТВЕТА: строго один JSON-объект: {"ok":true,"box":null} или {"ok":false,"box":[0.0,0.66,0.21,0.34]}"""

CAMCHECK_TOOL = {
    "name": "submit_cam",
    "description": "Проверка рамки вебки",
    "input_schema": {"type": "object", "properties": {
        "ok": {"type": "boolean"}, "box": {"type": ["array", "null"], "items": {"type": "number"}}},
        "required": ["ok"]},
}


# ------------------------------------------------------------------ кадры
def grab_frame(src: Path, t: float, w: int = TILE_W, h: int = TILE_H) -> np.ndarray | None:
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    p = subprocess.run([ffmpeg_bin(), "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1",
                        "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **kwargs)
    if len(p.stdout) < w * h * 3:
        return None
    return np.frombuffer(p.stdout[:w * h * 3], np.uint8).reshape(h, w, 3).copy()


def annotate(fr: np.ndarray, label: str, box: list[float] | None) -> np.ndarray:
    """Подпись времени, деления по краям (доли кадра) и жёлтая рамка вебки."""
    import cv2
    im = fr.copy()
    h, w = im.shape[:2]
    for k in range(1, 10):
        x, y = int(w * k / 10), int(h * k / 10)
        for (p1, p2, txt_pos) in (((x, 0), (x, 10), (x - 9, 22)), ((x, h - 10), (x, h), None),
                                  ((0, y), (10, y), (13, y + 5)), ((w - 10, y), (w, y), None)):
            cv2.line(im, p1, p2, (255, 255, 255), 2)
            cv2.line(im, p1, p2, (0, 0, 0), 1)
            if txt_pos:
                cv2.putText(im, f".{k}", txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
                cv2.putText(im, f".{k}", txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    if box:
        x, y, bw, bh = box
        cv2.rectangle(im, (int(x * w), int(y * h)), (int((x + bw) * w) - 1, int((y + bh) * h) - 1), (0, 230, 255), 2)
    cv2.rectangle(im, (w // 2 - 70, 0), (w // 2 + 70, 30), (0, 0, 0), -1)
    cv2.putText(im, label, (w // 2 - 62, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return im


def tiles(frames: list[np.ndarray], per: int = 4) -> list[bytes]:
    """Склейка по 4 кадра (2×2) в JPEG — так меньше картинок и токенов."""
    import cv2
    out = []
    for i in range(0, len(frames), per):
        group = frames[i:i + per]
        while len(group) < per:
            group.append(np.zeros_like(frames[0]))
        top = np.hstack(group[:2])
        bot = np.hstack(group[2:4])
        grid = np.vstack([top, bot])
        cv2.line(grid, (TILE_W, 0), (TILE_W, 2 * TILE_H), (40, 40, 40), 3)
        cv2.line(grid, (0, TILE_H), (2 * TILE_W, TILE_H), (40, 40, 40), 3)
        ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            out.append(buf.tobytes())
    return out


def _img_block(jpg: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(jpg).decode()}}


def sample_times(dur: float, keep: list[tuple[float, float]] | None, max_n: int = 16) -> list[float]:
    keep = keep or [(0.0, dur)]
    total = sum(b - a for a, b in keep)
    n = int(np.clip(round(total / 5.0), 4, max_n))
    n = (n + 3) // 4 * 4
    out = []
    for k in range(n):
        target = total * (k + 0.5) / n
        acc = 0.0
        for a, b in keep:
            if acc + (b - a) >= target:
                out.append(a + (target - acc))
                break
            acc += b - a
    return out


# ------------------------------------------------------------------ план
def _valid_rect(r) -> list[float] | None:
    try:
        x, y, w, h = [float(v) for v in r][:4]
    except (TypeError, ValueError):
        return None
    if w < 0.04 or h < 0.04:
        return None
    x, y = float(np.clip(x, 0, 0.98)), float(np.clip(y, 0, 0.98))
    w, h = min(w, 1 - x), min(h, 1 - y)
    return [round(x, 3), round(y, 3), round(w, 3), round(h, 3)]


def normalize_plan(data: dict, dur: float, min_len: float = 4.0) -> dict:
    shots = []
    for s in data.get("shots") or []:
        try:
            a, b = float(s.get("from", 0)), float(s.get("to", 0))
        except (TypeError, ValueError):
            continue
        lay = s.get("layout") if s.get("layout") in ("cam", "camwide", "split", "screen") else None
        if not lay:
            continue
        a, b = max(0.0, min(a, dur)), max(0.0, min(b, dur))
        if b <= a:
            continue
        focus = _valid_rect(s.get("focus")) if lay in ("split", "screen") else None
        if lay == "screen" and not focus:
            focus = [0.0, 0.0, 1.0, 1.0]
        if focus is not None and str(s.get("fit") or "").lower() == "whole":
            focus = focus + [1]  # показать область целиком (5-й элемент — флаг «не обрезать края»)
        shots.append({"t0": a, "t1": b, "layout": lay, "focus": focus, "why": str(s.get("why") or "")[:120]})
    shots.sort(key=lambda x: x["t0"])
    if not shots:
        return {"shots": [], "cuts": [], "cam_box": None}
    # без дыр и наложений: каждый шот до начала следующего
    shots[0]["t0"] = 0.0
    for i in range(len(shots) - 1):
        shots[i]["t1"] = shots[i + 1]["t0"]
    shots[-1]["t1"] = dur
    # короткие шоты сливаем с соседом
    changed = True
    while changed and len(shots) > 1:
        changed = False
        for i, s in enumerate(shots):
            if s["t1"] - s["t0"] < min_len:
                j = i - 1 if i > 0 else 1
                shots[j]["t0"], shots[j]["t1"] = min(shots[j]["t0"], s["t0"]), max(shots[j]["t1"], s["t1"])
                shots.pop(i)
                changed = True
                break
    merged = [shots[0]]
    for s in shots[1:]:
        if s["layout"] == merged[-1]["layout"] and s["focus"] == merged[-1]["focus"]:
            merged[-1]["t1"] = s["t1"]
        else:
            merged.append(s)
    cuts = []
    for c in data.get("cuts") or []:
        try:
            a, b = float(c[0]), float(c[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b - a >= 1.0 and 0 <= a < b <= dur:
            cuts.append([round(a, 2), round(b, 2)])
    if sum(b - a for a, b in cuts) > 0.3 * dur:
        cuts = []
    return {"shots": merged, "cuts": cuts, "cam_box": _valid_rect(data.get("cam_box")) if data.get("cam_box") else None}


def cached_plan(cache: Path | None, dur: float) -> dict | None:
    """Готовое решение режиссёра из прошлого монтажа (для перемонтажа без запросов к Claude)."""
    old = read_json(cache) if cache else None
    if not old or not old.get("plan") or not old["plan"].get("shots"):
        return None
    if abs(float(old.get("dur") or old["plan"]["shots"][-1]["t1"]) - dur) > 1.0:
        return None  # клип с тех пор ужали/расширили — старый план не подходит
    return old["plan"]


def plan_clip(src: Path, offset: float, dur: float, words_rel: list[dict], keep: list | None, li: dict,
              cfg: dict, workdir: Path, usage: dict, about: str = "", notes: str = "",
              cache: Path | None = None) -> dict | None:
    """Решение режиссёра для одного клипа (время — от начала клипа, исходная шкала). None — не получилось."""
    from .llm import llm_vision
    rc = cfg.get("director") or {}
    box = li.get("cam_box")
    times = sample_times(dur, keep, int(rc.get("max_frames", 12)))
    key = hashlib.md5(json.dumps([VERSION, round(dur, 1), [round(t, 1) for t in times], box, about[:200], notes[:400],
                                  rc.get("model")], ensure_ascii=False).encode()).hexdigest()[:12]
    if cache:
        old = read_json(cache)
        if old and old.get("key") == key:
            return old["plan"]
    frames = []
    for t in times:
        fr = grab_frame(src, offset + t)
        if fr is not None:
            frames.append(annotate(fr, f"t={t:.1f}s", box))
    if len(frames) < 2:
        return None
    imgs = tiles(frames)
    lines, cur, cur_t, last = [], [], None, None
    for w in words_rel:
        if cur and (w["s"] - last > 0.8 or len(cur) >= 14 or cur[-1].endswith((".", "!", "?"))):
            lines.append(f"[{cur_t:.1f}] " + " ".join(cur))
            cur = []
        if not cur:
            cur_t = w["s"]
        cur.append(w["w"])
        last = w["e"]
    if cur:
        lines.append(f"[{cur_t:.1f}] " + " ".join(cur))
    cut_txt = ""
    if keep and (len(keep) > 1 or keep[0][0] > 0.5):
        gaps = [(round(b0, 1), round(a1, 1)) for (a0, b0), (a1, b1) in zip(keep, keep[1:]) if a1 - b0 > 1.0]
        if gaps:
            cut_txt = f"\nУже вырезано (паузы/вода): {gaps}"
    extra = ""
    if notes.strip():
        extra = "\nПОЖЕЛАНИЯ АВТОРА ПО МОНТАЖУ ЭТОГО КАНАЛА (по его прошлым замечаниям):\n" + notes.strip()[:1500]
    system = DIRECTOR_SYSTEM.format(extra=extra)
    text = (f"Клип длиной {dur:.1f} с. {about}{cut_txt}\n"
            f"Кадры по порядку: {', '.join(f't={t:.1f}' for t in times[:len(frames)])} (по 4 на картинке: "
            f"слева направо, сверху вниз).\n\nРасшифровка:\n" + ("\n".join(lines) or "(речи нет)"))
    blocks = [_img_block(j) for j in imgs] + [{"type": "text", "text": text}]
    data = llm_vision(system, blocks, cfg, usage, workdir, DIRECTOR_TOOL, DIRECTOR_JSON,
                      cli_model=rc.get("model") or cfg["llm"].get("cli_model"))
    plan = normalize_plan(data, dur, float(rc.get("min_shot_seconds", 4.0)))
    if not plan["shots"]:
        return None
    if cache:
        write_json(cache, {"key": key, "plan": plan, "dur": round(dur, 2)})
    return plan


def apply_plan(li: dict, plan: dict) -> dict:
    """Шоты режиссёра → сегменты раскладки (с учётом того, что вообще можно показать)."""
    li = dict(li)
    segs = []
    for s in plan["shots"]:
        lay = s["layout"]
        if lay in ("cam", "camwide", "split") and not li.get("cam"):
            lay = "screen"
        if lay == "camwide" and not li.get("camwide_rect"):
            lay = "cam"  # вебка и так показывается целиком
        if lay == "cam" and not li.get("camcrop"):
            lay = "split"
        segs.append({"t0": s["t0"], "t1": s["t1"], "layout": lay,
                     "focus": s.get("focus") if lay in ("split", "screen") else None})
    if segs:
        li["segments"] = segs
        li["layout"] = max(segs, key=lambda x: x["t1"] - x["t0"])["layout"]
        li["director"] = True
    return li


# ------------------------------------------------------------- проверка вебки
def _snap_box(box: list[float], frames: list) -> list[float]:
    """Подгоняет рамку от Claude к настоящим линиям рамки вебки (±3% кадра), если они есть."""
    from .render import _EdgeStack
    if len(frames) < 4:
        return box
    es = _EdgeStack(frames)
    w, h = es.w, es.h
    L, T = box[0] * w, box[1] * h
    R, B = (box[0] + box[2]) * w, (box[1] + box[3]) * h
    d = int(0.03 * max(w, h))

    def best_h(y0):
        cand = [(es.h_score(y, L, R), y) for y in range(max(0, int(y0) - d), min(h, int(y0) + d + 1))]
        sc, y = max(cand) if cand else (0, y0)
        return y if sc > 0.6 else y0

    def best_v(x0):
        cand = [(es.v_score(x, T, B), x) for x in range(max(0, int(x0) - d), min(w, int(x0) + d + 1))]
        sc, x = max(cand) if cand else (0, x0)
        return x if sc > 0.6 else x0
    T2, B2 = best_h(T), best_h(B)
    L2, R2 = best_v(L), best_v(R)
    if R2 - L2 < 0.5 * (R - L) or B2 - T2 < 0.5 * (B - T):
        return box
    return [round(L2 / w, 4), round(T2 / h, 4), round((R2 - L2) / w, 4), round((B2 - T2) / h, 4)]


def verify_cam(frames: list, box: list[float], cfg: dict, workdir: Path, usage: dict) -> dict:
    """Один запрос на стрим: Claude смотрит 4 кадра из разных клипов с нарисованной рамкой вебки.
    Возвращает {"ok": bool, "box": исправленная рамка или None}."""
    from .llm import llm_vision
    import cv2
    pick = frames[:: max(1, len(frames) // 4)][:4]
    if len(pick) < 2:
        return {"ok": True, "box": None}
    shown = []
    for i, (_, fr) in enumerate(pick):
        im = cv2.resize(fr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        shown.append(annotate(im, f"#{i + 1}", box))
    blocks = [_img_block(j) for j in tiles(shown)] + [{"type": "text", "text": "Проверь жёлтую рамку вебки на этих кадрах."}]
    rc = cfg.get("director") or {}
    data = llm_vision(CAMCHECK_SYSTEM, blocks, cfg, usage, workdir, CAMCHECK_TOOL, CAMCHECK_JSON,
                      cli_model=rc.get("model") or cfg["llm"].get("cli_model"))
    if data.get("ok"):
        return {"ok": True, "box": None}
    nb = _valid_rect(data.get("box"))
    if nb:
        nb = _snap_box(nb, frames)
    return {"ok": False, "box": nb}


def enabled(cfg: dict) -> bool:
    rc = cfg.get("director") or {}
    return bool(rc.get("enabled", True)) and cfg["llm"]["backend"] in ("api", "claude-cli")
