"""Монтаж вертикального клипа 9:16: раскладка (вебка + игра), субтитры по словам, запикивание мата.

Раскладки:
  split — вебка сверху, геймплей снизу (если в кадре найдена небольшая стабильная вебка)
  crop  — кадрирование по лицу (стрим «общение», лицо крупно)
  blur  — исходный кадр по центру на размытом фоне (универсальный запасной вариант)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .profanity import ProfanityFilter
from .util import ROOT_DIR, ffmpeg_bin, ffprobe_video, log, run

YUNET = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
# версия логики раскладки: клипы, смонтированные старой версией, после обновления перемонтируются сами
# (без запросов к Claude). 2 — в кадре показываются все участники (соведущий на вебке, гость, второй стример)
LAYOUT_VERSION = 2


# ------------------------------------------------------------ face / layout
def _sample_keyframes(path: Path, offset: float, dur: float, sw: int, sh: int) -> list[tuple[float, np.ndarray]]:
    """Кадры клипа для анализа. Декодируются только ключевые кадры (обычно раз в 2 с) — это быстро даже
    для 1440p/HEVC. Возвращает [(время от начала клипа, кадр BGR 640px)]."""
    import re as _re
    import subprocess
    w = 640
    h = int(round(sh * w / sw / 2)) * 2
    cmd = [ffmpeg_bin(), "-v", "info", "-hide_banner", "-skip_frame", "nokey", "-ss", f"{offset:.3f}",
           "-t", f"{dur:.3f}", "-i", str(path), "-map", "0:v:0", "-vsync", "0",
           "-vf", f"showinfo,scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    size = w * h * 3
    n = len(p.stdout) // size
    times = [float(x) for x in _re.findall(r"pts_time:\s*([-\d.]+)", p.stderr.decode("utf-8", "replace"))]
    frames = []
    for i in range(n):
        t = times[i] if i < len(times) and len(times) == n else dur * (i + 0.5) / max(1, n)
        frames.append((max(0.0, t), np.frombuffer(p.stdout[i * size:(i + 1) * size], np.uint8).reshape(h, w, 3)))
    if n < max(3, dur / 3.5):  # редкие ключевые кадры — добираем снимками в промежутках (для слежения за лицом)
        have = sorted(t for t, _ in frames)
        want = [dur * (k + 0.5) / min(40, max(6, int(dur / 2))) for k in range(min(40, max(6, int(dur / 2))))]
        for t in want:
            if any(abs(t - x) < 1.2 for x in have):
                continue
            q = subprocess.run([ffmpeg_bin(), "-v", "error", "-ss", f"{offset + t:.3f}", "-i", str(path),
                                "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **kwargs)
            if len(q.stdout) >= size:
                frames.append((t, np.frombuffer(q.stdout[:size], np.uint8).reshape(h, w, 3)))
        frames.sort(key=lambda x: x[0])
    return frames


def _detect_faces(frames: list[tuple[float, np.ndarray]]) -> list[list[tuple]]:
    """Для каждого кадра: список лиц (cx, cy, w, h, score) в долях кадра."""
    import cv2
    out: list[list[tuple]] = []
    if not frames or not YUNET.exists():
        return [[] for _ in frames]
    h, w = frames[0][1].shape[:2]
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.72, 0.3, 50)
    for _, fr in frames:
        _, res = det.detect(fr)
        faces = []
        if res is not None:
            for f in res:
                x, y, fw, fh = [float(v) for v in f[:4]]
                if fw < 10:  # совсем мелкие лица (толпа, превью) не интересны
                    continue
                faces.append(((x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h, float(f[-1])))
        out.append(faces)
    return out


def _track_camera(times: list[float], faces: list[list[tuple]], crop_w_frac: float,
                  dead_frac: float = 0.18) -> list[tuple[float, float]]:
    """Плавная «виртуальная камера» по главному лицу: держит его в кадре, но не дёргается от
    каждого движения (мёртвая зона). Возвращает опорные точки (t, центр x в долях кадра)."""
    pts: list[tuple[float, float]] = []
    prev = None
    for t, fs in zip(times, faces):
        if not fs:
            continue
        if prev is None:
            main = max(fs, key=lambda f: f[2] * f[3])
        else:  # то же лицо, что и раньше (ближайшее), с учётом размера
            main = min(fs, key=lambda f: abs(f[0] - prev) - f[2])
        prev = main[0]
        pts.append((t, main[0]))
    if not pts:
        return []
    dead = crop_w_frac * dead_frac
    cam = pts[0][1]
    keys = [(0.0, cam)]
    for t, x in pts:
        diff = x - cam
        if abs(diff) > dead:
            cam += diff - math.copysign(dead, diff)
            keys.append((t, cam))
    # убираем лишние точки на прямых участках
    out = [keys[0]]
    for k in keys[1:]:
        if abs(k[1] - out[-1][1]) > 0.004:
            out.append(k)
    return out


def screen_is_empty(fr: np.ndarray, cam_rel: list[float]) -> bool:
    """Основной экран (кроме области вебки) пустой: чёрный или однотонный БЕЗ деталей.
    Белая страница сайта с текстом — не пустая (там много мелких деталей/краёв)."""
    import cv2
    h, w = fr.shape[:2]
    x, y, cw, ch = cam_rel
    pad = 0.08
    mask = np.ones((h, w), bool)
    mask[max(0, int((y - pad) * h)):int((y + ch + pad) * h), max(0, int((x - pad) * w)):int((x + cw + pad) * w)] = False
    g = fr.mean(axis=2).astype(np.uint8)
    edges = cv2.Canny(g, 60, 140) > 0
    gm = g[mask]
    edge = float(edges[mask].mean())
    dark = float((gm < 32).mean())
    flat = float((np.abs(gm.astype(np.int16) - int(np.median(gm))) < 14).mean())
    return (dark >= 0.8 and edge < 0.012) or (flat >= 0.9 and edge < 0.004)


def screen_activity(frames: list, cam_rel: list[float]) -> dict:
    if not frames:
        return {}
    flags = [screen_is_empty(fr, cam_rel) for _, fr in frames]
    return {"empty_share": round(sum(flags) / len(flags), 2), "empty": sum(flags) >= 0.7 * len(flags)}


def cam_box(frames: list, cx: float, cy: float) -> list[float] | None:
    """Границы картинки вебки на пустом экране: связная область, отличающаяся от фона, вокруг лица."""
    import cv2
    if not frames:
        return None
    fr = frames[len(frames) // 2][1]
    g = fr.mean(axis=2)
    bg = float(np.median(g))
    m = (np.abs(g - bg) > 14).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    h, w = g.shape
    px, py = int(cx * w), int(cy * h)
    i = lab[min(h - 1, py), min(w - 1, px)]
    if i == 0:
        return None
    x, y, bw, bh = st[i][:4]
    if bw * bh > 0.6 * w * h:  # область почти весь кадр — это не рамка вебки
        return None
    return [float(x / w), float(y / h), float(bw / w), float(bh / h)]


def _smooth_modes(times: list[float], modes: list[str], dur: float, min_len: float = 4.0) -> list[dict]:
    """Режимы по кадрам → отрезки [t0, t1, layout] без дребезга (короткие отрезки сливаются с соседями)."""
    if not modes:
        return []
    # сглаживание большинством в окне из 3 кадров
    sm = []
    for i in range(len(modes)):
        win = modes[max(0, i - 1): i + 2]
        sm.append(max(set(win), key=win.count) if len(win) == 3 and win.count(modes[i]) == 1 else modes[i])
    bounds = [0.0] + [(times[i] + times[i + 1]) / 2 for i in range(len(times) - 1)] + [dur]
    segs: list[dict] = []
    for i, m in enumerate(sm):
        if segs and segs[-1]["layout"] == m:
            segs[-1]["t1"] = bounds[i + 1]
        else:
            segs.append({"t0": bounds[i], "t1": bounds[i + 1], "layout": m})
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, sgm in enumerate(segs):
            if sgm["t1"] - sgm["t0"] < min_len:
                j = i - 1 if i > 0 else i + 1
                if 0 < i < len(segs) - 1 and (segs[i + 1]["t1"] - segs[i + 1]["t0"]) > (segs[i - 1]["t1"] - segs[i - 1]["t0"]):
                    j = i + 1
                segs[j]["t0"] = min(segs[j]["t0"], sgm["t0"])
                segs[j]["t1"] = max(segs[j]["t1"], sgm["t1"])
                segs.pop(i)
                changed = True
                break
        # склеиваем соседей с одинаковой раскладкой
        k = 1
        while k < len(segs):
            if segs[k]["layout"] == segs[k - 1]["layout"]:
                segs[k - 1]["t1"] = segs[k]["t1"]
                segs.pop(k)
            else:
                k += 1
    return segs


# ------------------------------------------------------- вебка на весь стрим
def _hrun(arr: np.ndarray, start: int, thr: float, gap: int = 4) -> tuple[int, int]:
    """Границы «полосы» сильных значений вокруг start (с допуском разрывов до gap пикселей)."""
    n = len(arr)
    l = r = max(0, min(n - 1, start))
    for step, lim in ((-1, -1), (1, n)):
        miss, x = 0, l if step < 0 else r
        while True:
            x += step
            if x == lim:
                break
            if arr[x] > thr:
                miss = 0
                if step < 0:
                    l = x
                else:
                    r = x
            else:
                miss += 1
                if miss > gap:
                    break
    return l, r


class _EdgeStack:
    """Края кадров (по многим кадрам стрима) с быстрыми суммами вдоль строк/столбцов.
    Рамка вебки — прямая линия, которая есть почти в КАЖДОМ кадре (человек и экран двигаются, рамка — нет)."""

    def __init__(self, frames: list):
        import cv2
        grays = [f.mean(axis=2).astype(np.float32) for _, f in frames]
        self.h, self.w = grays[0].shape
        gy = np.stack([np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) for g in grays]) > 40
        gx = np.stack([np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) for g in grays]) > 40
        # линия может быть смазана на пиксель — берём полосу в 3 пикселя
        gy = gy | np.roll(gy, 1, axis=1) | np.roll(gy, -1, axis=1)
        gx = gx | np.roll(gx, 1, axis=2) | np.roll(gx, -1, axis=2)
        self.cy = np.concatenate([np.zeros(gy.shape[:2] + (1,), np.int16), np.cumsum(gy, axis=2, dtype=np.int16)], axis=2)
        self.cx = np.concatenate([np.zeros((gx.shape[0], 1, gx.shape[2]), np.int16), np.cumsum(gx, axis=1, dtype=np.int16)], axis=1)

    def h_score(self, y: int, x0: float, x1: float) -> float:
        """Доля длины горизонтальной линии y на [x0, x1], «типичная» по кадрам. Край кадра = 1."""
        if y <= 0 or y >= self.h - 1:
            return 1.0
        a, b = int(max(0, x0)), int(min(self.w, x1))
        if b - a < 4:
            return 0.0
        return float(np.median((self.cy[:, y, b] - self.cy[:, y, a]) / (b - a)))

    def v_score(self, x: int, y0: float, y1: float) -> float:
        if x <= 0 or x >= self.w - 1:
            return 1.0
        a, b = int(max(0, y0)), int(min(self.h, y1))
        if b - a < 4:
            return 0.0
        return float(np.median((self.cx[:, b, x] - self.cx[:, a, x]) / (b - a)))

    def box_score(self, L: int, T: int, R: int, B: int) -> tuple[float, list[float]]:
        sides = [self.h_score(T, L, R), self.h_score(B, L, R), self.v_score(L, T, B), self.v_score(R, T, B)]
        return min(sides) + 0.25 * float(np.mean(sides)), sides


def _box_score_rel(es: _EdgeStack, box: list[float]) -> float:
    x, y, w, h = box
    L, T = int(round(x * es.w)), int(round(y * es.h))
    R, B = int(round((x + w) * es.w)), int(round((y + h) * es.h))
    best = 0.0
    for d in (-2, 0, 2):  # рамка могла быть сохранена с отступом внутрь
        best = max(best, es.box_score(L + (d if L > 0 else 0), T + (d if T > 0 else 0),
                                      R - (d if R < es.w else 0), B - (d if B < es.h else 0))[0])
    return best


def cam_rect_from_edges(frames: list, cx: float, cy: float, fw: float, es: "_EdgeStack | None" = None,
                        with_score: bool = False):
    """Точные границы картинки вебки [x, y, w, h] в долях кадра или None.
    Перебираем прямоугольники из устойчивых линий вокруг лица (и краёв кадра) и выбираем тот, у которого
    ВСЕ четыре стороны — настоящие линии рамки, а пропорции похожи на камеру (16:9 или 4:3)."""
    import itertools
    if len(frames) < 4:
        return (None, 0.0) if with_score else None
    es = es or _EdgeStack(frames)
    h, w = es.h, es.w
    px, py, f = cx * w, cy * h, fw * w
    span = 1.3 * f

    def cands(lo, hi, fn, lim):
        out = []
        for v in range(int(max(1, lo)), int(min(lim - 1, hi))):
            sc = fn(v)
            if sc > 0.6:
                out.append((sc, v))
        out.sort(reverse=True)
        picked: list[int] = []
        for sc, v in out:
            if all(abs(v - p) > 2 for p in picked):
                picked.append(v)
            if len(picked) >= 8:
                break
        return picked
    tops = [0] + cands(py - 6 * f, py - 0.9 * f, lambda y: es.h_score(y, px - span, px + span), h)
    bots = [h] + cands(py + 0.9 * f, py + 7 * f, lambda y: es.h_score(y, px - span, px + span), h)
    lefts = [0] + cands(px - 7 * f, px - 0.9 * f, lambda x: es.v_score(x, py - span, py + span), w)
    rights = [w] + cands(px + 0.9 * f, px + 7 * f, lambda x: es.v_score(x, py - span, py + span), w)
    best = None
    for T, B, L, R in itertools.product(tops, bots, lefts, rights):
        bw, bh = R - L, B - T
        if bw < 3.5 * f or bw > 10 * f or bh < 2.5 * f:
            continue
        asp = bw / bh
        if not (1.15 < asp < 2.0):
            continue
        if not (L < px - 0.8 * f and R > px + 0.8 * f and T < py - 0.8 * f and B > py + 0.8 * f):
            continue
        sc, _ = es.box_score(L, T, R, B)
        sc -= 0.5 * min(abs(asp - 16 / 9), abs(asp - 4 / 3) + 0.1)
        if best is None or sc > best[0]:
            best = (sc, L, T, R, B)
    if best is None or best[0] < 0.55:
        return (None, 0.0) if with_score else None
    sc, L, T, R, B = best
    # отступ внутрь от рамки, чтобы не попала сама линия
    L, T = (L + 2 if L > 0 else 0), (T + 2 if T > 0 else 0)
    R, B = (R - 2 if R < w else w), (B - 2 if B < h else h)
    box = [L / w, T / h, (R - L) / w, (B - T) / h]
    return (box, sc) if with_score else box


def stream_cam(samples: list[tuple[list, list]], prior: dict | None = None) -> dict | None:
    """Где вебка стримера — по ВСЕМ клипам стрима сразу. Вебка стоит на одном месте весь стрим, а лица
    из роликов, которые стример смотрит, в разных клипах разные. samples: [(кадры, лица по кадрам)] на клип.
    prior — вебка, найденная у этого стримера раньше (небольшой бонус, если совпадает)."""
    n = len(samples)
    if not n:
        return None
    clusters: list[dict] = []
    for ci, (_frames, faces) in enumerate(samples):
        for fs in faces:
            for f in fs:
                if f[2] >= 0.12:
                    continue
                for cl in clusters:
                    if abs(cl["cx"] - f[0]) < 0.035 and abs(cl["cy"] - f[1]) < 0.06 and 0.6 < f[2] / cl["fw"] < 1.6:
                        k = cl["n"]
                        cl["cx"], cl["cy"], cl["fw"] = ((cl["cx"] * k + f[0]) / (k + 1), (cl["cy"] * k + f[1]) / (k + 1),
                                                        (cl["fw"] * k + f[2]) / (k + 1))
                        cl["n"] += 1
                        cl["per"][ci] = cl["per"].get(ci, 0) + 1
                        break
                else:
                    clusters.append({"cx": f[0], "cy": f[1], "fw": f[2], "n": 1, "per": {ci: 1}})
    if not clusters:
        return None
    best, best_score = None, -1.0
    for cl in clusters:
        present = sum(1 for ci, cnt in cl["per"].items() if cnt >= max(1, 0.1 * len(samples[ci][1])))
        score = present + 0.002 * cl["n"]
        if prior and abs(prior["face"][0] - cl["cx"]) < 0.04 and abs(prior["face"][1] - cl["cy"]) < 0.06:
            score += max(1.0, 0.5 * n)
            cl["prior"] = True
        cl["present"] = present
        if score > best_score:
            best, best_score = cl, score
    need = 2 if n >= 2 else 1
    if best["present"] < max(need, 0.5 * n) and not best.get("prior"):
        return None
    if n == 1 and not best.get("prior") and best["n"] < 0.3 * len(samples[0][1]):
        return None
    frames = [fr for fs, _ in samples for fr in fs[:: max(1, len(fs) // 24)]][:72]
    box = None
    try:
        es = _EdgeStack(frames)
        box, sc = cam_rect_from_edges(frames, best["cx"], best["cy"], best["fw"], es=es, with_score=True)
        if prior and best.get("prior") and prior.get("box"):
            # рамка, найденная раньше, — если она по-прежнему лучше совпадает с линиями в кадре, оставляем её
            px_, py_, _ = best["cx"], best["cy"], best["fw"]
            pb = prior["box"]
            if pb[0] <= px_ <= pb[0] + pb[2] and pb[1] <= py_ <= pb[1] + pb[3]:
                sp = _box_score_rel(es, pb)
                if box is None or sp >= sc - 0.05:
                    box = pb
    except Exception as e:
        log.warning("Рамка вебки не найдена: %s", e)
        if prior and best.get("prior") and prior.get("box"):
            box = prior["box"]
    res = {"face": [round(best["cx"], 4), round(best["cy"], 4), round(best["fw"], 4)],
           "box": [round(v, 4) for v in box] if box else None, "clips": best["present"], "of": n,
           "updated": __import__("time").time()}
    # второй участник со СВОЕЙ вебкой (второй стример, гость): маленькое лицо на одном месте почти во всех
    # клипах, вне рамки главной вебки. Лица из роликов на экране так не держатся — в разных клипах они разные
    others = []
    if n >= 2:
        for cl in clusters:
            if cl is best or cl.get("present", 0) < max(2, 0.6 * n) or not (0.5 < cl["fw"] / best["fw"] < 2.0):
                continue
            if box and box[0] - 0.01 <= cl["cx"] <= box[0] + box[2] + 0.01 and box[1] - 0.01 <= cl["cy"] <= box[1] + box[3] + 0.01:
                continue  # сидит на той же вебке — его найдёт раскладка клипа
            if any(abs(cl["cx"] - o["face"][0]) < 0.05 and abs(cl["cy"] - o["face"][1]) < 0.08 for o in others):
                continue
            ob = None
            try:
                ob = cam_rect_from_edges(frames, cl["cx"], cl["cy"], cl["fw"])
            except Exception:
                pass
            others.append({"face": [round(cl["cx"], 4), round(cl["cy"], 4), round(cl["fw"], 4)],
                           "box": [round(v, 4) for v in ob] if ob else None, "clips": cl["present"]})
    if others:
        res["others"] = others[:2]
        log.info("Ещё вебки участников: %s", [o["face"] for o in res["others"]])
    log.info("Вебка стримера: лицо %s, рамка %s (в %d из %d клипов)", res["face"], res["box"], best["present"], n)
    return res


def sample_for_cam(path: Path, offset: float, dur: float) -> tuple[list, list]:
    """Кадры и лица одного клипа для поиска вебки по всему стриму (не больше ~16 кадров)."""
    info = ffprobe_video(path)
    sw, sh = int(info.get("width") or 1920), int(info.get("height") or 1080)
    frames = _sample_keyframes(path, offset, dur, sw, sh)
    frames = frames[:: max(1, len(frames) // 16)]
    return frames, _detect_faces(frames)


def _default_cam_box(cx: float, cy: float, fw: float, sw: int, sh: int) -> list[float]:
    """Рамку вебки не нашли — берём область 16:9 вокруг лица (типичная вебка ≈ 5.5 ширин лица)."""
    bw = min(1.0, fw * 5.6)
    bh = min(1.0, bw * sw / sh * 9 / 16)
    return [float(np.clip(cx - bw / 2, 0, 1 - bw)), float(np.clip(cy - bh * 0.45, 0, 1 - bh)), bw, bh]


def top_height(box: list[float], sw: int, sh: int, cfg: dict) -> int:
    """Высота зоны вебки в «вебка + экран»: под пропорции самой вебки, чтобы она вошла ЦЕЛИКОМ."""
    W = int(cfg["render"]["width"])
    bw, bh = box[2] * sw, box[3] * sh
    if bw <= 0 or bh <= 0:
        return cam_height(cfg)
    lo, hi = int(cfg["render"].get("cam_height_min", 560)), int(cfg["render"].get("cam_height_max", 1100))
    return int(np.clip(W * bh / bw, lo, hi)) // 2 * 2


def _screen_motion(frames: list, cam_rel: list[float]) -> list[dict]:
    """Что происходит на экране (без области вебки) между соседними кадрами: доля изменившейся площади
    самого большого «живого» участка (видео, игра) и по столбцам — где именно движение."""
    import cv2
    if len(frames) < 2:
        return [{"blob": 0.0, "cols": None} for _ in frames]
    small = []
    for _, fr in frames:
        g = cv2.resize(fr.mean(axis=2).astype(np.uint8), (192, 108), interpolation=cv2.INTER_AREA)
        small.append(cv2.GaussianBlur(g, (3, 3), 0).astype(np.int16))
    h, w = small[0].shape
    x, y, cw, ch = cam_rel
    pad = 0.03
    mask = np.ones((h, w), bool)
    mask[max(0, int((y - pad) * h)):int((y + ch + pad) * h) + 1, max(0, int((x - pad) * w)):int((x + cw + pad) * w) + 1] = False
    out = []
    for i in range(len(small)):
        j = i - 1 if i > 0 else 1
        d = (np.abs(small[i] - small[j]) > 18) & mask
        d8 = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(d8)
        blob = 0.0
        for k in range(1, n):
            bx, by, bw, bh, area = st[k]
            if bw >= 0.18 * w and bh >= 0.12 * h:  # широкий участок — видео/игра, а не лента чата
                blob = max(blob, area / float(mask.sum()))
        out.append({"blob": blob, "cols": d.sum(axis=0).astype(np.float32)})
    return out


def _cam_modes(times: list[float], frames: list, faces: list, cam_rel: list[float], cam_face: tuple,
               words: list[dict] | None, dur: float, cfg: dict, hint: dict | None = None) -> tuple[list[str], dict]:
    """Режим для каждого кадра, когда у стримера есть вебка. Лицо стримера видно ВСЕГДА:
    split — вебка сверху + экран снизу (на экране что-то происходит и стример это комментирует);
    cam   — вебка на весь кадр (экран стоит/пустой, а стример долго рассказывает)."""
    r = cfg["render"]
    mot = _screen_motion(frames, cam_rel)
    empty = [screen_is_empty(fr, cam_rel) for _, fr in frames]
    act_thr = float(r.get("screen_active", 0.05))
    active = [(not e) and m["blob"] >= act_thr for e, m in zip(empty, mot)]
    cx, cy, fw = cam_face
    on_cam = [any(abs(f[0] - cx) < 0.06 and abs(f[1] - cy) < 0.09 for f in fs) for fs in faces]

    def talking(t: float) -> bool:
        return bool(words) and any(w["s"] - 1.5 <= t <= w["e"] + 1.5 for w in words)
    n = len(frames)
    info = {"screen_active": round(sum(active) / max(1, n), 2)}
    if not r.get("dynamic_layout", True):
        return ["split"] * n, info
    visual = (hint or {}).get("visual")
    spans = (hint or {}).get("spans") or []
    info["visual"] = visual
    # Claude по смыслу сказанного решил, нужен ли экран: это надёжнее, чем одно движение на экране
    if visual == "talk":
        return ["cam" if (on_cam[i] or not any(on_cam)) else "split" for i in range(n)], info
    if visual == "screen":
        return ["split"] * n, info
    if visual == "mixed" and spans:
        return [("split" if any(a - 1.0 <= times[i] <= b + 1.0 for a, b in spans)
                 else ("cam" if (on_cam[i] or not any(on_cam)) else "split")) for i in range(n)], info
    if n and sum(active) <= 0.15 * n:
        # экран весь клип стоит (OBS, пустой рабочий стол, статичная страница) — только вебка
        return ["cam" if (on_cam[i] or not any(on_cam)) else "split" for i in range(n)], info
    modes = ["split" if a else "cam?" for a in active]
    # «cam» — только длинные отрезки без действия на экране, где стример говорит, и не в самом начале:
    # первые секунды зритель должен увидеть, о чём речь
    min_cam = float(r.get("cam_min_seconds", 7.0))
    intro = float(r.get("intro_screen_seconds", 8.0))
    i = 0
    while i < n:
        if modes[i] != "cam?":
            i += 1
            continue
        j = i
        while j < n and modes[j] == "cam?":
            j += 1
        t0 = times[i] if i else 0.0
        t1 = times[j] if j < n else dur
        span = [k for k in range(i, j) if times[k] >= intro]
        talk = sum(1 for k in span if talking(times[k]))
        ok = (t1 - max(t0, intro) >= min_cam and span and talk >= 0.5 * len(span)
              and sum(on_cam[k] for k in span) >= 0.6 * len(span))
        for k in range(i, j):
            modes[k] = "cam" if ok and times[k] >= intro else "split"
        i = j
    return modes, info


def _screen_focus(mot: list[dict], sw: int, gw: int) -> int:
    """Где на экране главное действие (идущее видео, игра): окно ширины gw с наибольшим движением."""
    cols = [m["cols"] for m in mot if m.get("cols") is not None and m.get("blob", 0) > 0]
    if not cols or gw >= sw:
        return max(0, (sw - gw) // 2)
    energy = np.sum(cols, axis=0)
    w = len(energy)
    win = max(1, int(round(gw / sw * w)))
    if energy.sum() <= 0:
        return (sw - gw) // 2
    cs = np.concatenate([[0], np.cumsum(energy)])
    sums = cs[win:] - cs[:-win]
    k = int(np.argmax(sums))
    # без резких перекосов: если движение равномерное — по центру
    if sums[k] < 1.15 * sums[(w - win) // 2]:
        return (sw - gw) // 2
    return int(np.clip(k / w * sw, 0, sw - gw)) // 2 * 2


def detect_layout(path: Path, offset: float, dur: float, cfg: dict, words: list[dict] | None = None,
                  hint: dict | None = None) -> dict:
    """Раскладка кадра — по каждому ключевому кадру клипа, поэтому внутри одного клипа она может меняться.
    Если у стримера есть вебка — его лицо в клипе видно всегда: «вебка + экран», пока на экране что-то
    происходит, и «только вебка», когда экран стоит, а стример долго рассказывает.
    words — слова клипа (время от начала клипа), чтобы понимать, когда стример говорит."""
    r = cfg["render"]
    info = ffprobe_video(path)
    sw, sh = int(info.get("width") or 1920), int(info.get("height") or 1080)
    W, H = int(r["width"]), int(r["height"])
    res: dict = {"src_w": sw, "src_h": sh, "layout": "blur"}
    want = r["layout"]

    try:
        frames = _sample_keyframes(path, offset, dur, sw, sh)
        faces = _detect_faces(frames)
    except Exception as e:  # детекция не должна ломать монтаж
        log.warning("Детекция лица недоступна: %s", e)
        frames, faces = [], []
    n = len(frames)
    times = [t for t, _ in frames]
    res["face_frames"] = f"{sum(1 for f in faces if f)}/{n}"

    # --- вебка поверх экрана: маленькое лицо, стоящее на одном месте
    cam_face = None
    box = None
    scam = r.get("stream_cam") if not r.get("cam_rect") else None
    if r.get("cam_rect"):
        x, y, w, h = r["cam_rect"]
        box = [x, y, w, h]
        cam_face = (x + w / 2, y + h * 0.45, w / 5.6)
        inside = [f for fs in faces for f in fs if x <= f[0] <= x + w and y <= f[1] <= y + h]
        if inside:
            cam_face = (float(np.median([f[0] for f in inside])), float(np.median([f[1] for f in inside])),
                        float(np.median([f[2] for f in inside])))
    elif scam:
        cam_face = tuple(scam["face"])
        box = scam.get("box")
    else:
        flat = [(i, f) for i, fs in enumerate(faces) for f in fs if f[2] < 0.11]
        best, support = None, 0
        for i, f in flat:
            sup = {j for j, g in flat if abs(g[0] - f[0]) < 0.05 and abs(g[1] - f[1]) < 0.07 and 0.6 < g[2] / f[2] < 1.6}
            if len(sup) > support:
                best, support = f, len(sup)
        if best is not None and support >= max(2, 0.3 * n):
            cl = [g for _, g in flat if abs(g[0] - best[0]) < 0.05 and abs(g[1] - best[1]) < 0.07]
            cx, cy = float(np.median([g[0] for g in cl])), float(np.median([g[1] for g in cl]))
            fw = float(np.median([g[2] for g in cl]))
            if (cx < 0.33 or cx > 0.67) or (cy < 0.3 or cy > 0.7):
                cam_face = (cx, cy, fw)
                try:
                    box = cam_rect_from_edges(frames, cx, cy, fw) or cam_box(frames, cx, cy)
                except Exception:
                    box = None
    cam_rel = None
    if cam_face:
        cx, cy, fw = cam_face
        if not box:
            box = _default_cam_box(cx, cy, fw, sw, sh)
        if r.get("cam_center", True):
            # лицо — по центру зоны вебки и крупно (как в «кадре по лицу»), камера следит за лицом
            top = cam_height(cfg)
            cam_rel = _split_cam_crop(box, cam_face, sw, sh, W / top, float(r.get("cam_zoom", 1.0)))
        else:
            # вебка целиком, зона подстраивается под её пропорции
            top = top_height(box, sw, sh, cfg)
            cam_rel = _fit_aspect(box, cam_face, sw, sh, W / top)
        x, y, w, h = cam_rel
        res["cam"] = [int(x * sw), int(y * sh), int(w * sw) // 2 * 2, int(h * sh) // 2 * 2]
        res["top_h"] = top
        res.update(face_cx=int(cx * sw), face_cy=int(cy * sh), face_w=fw)
        res["cam_box"] = box
        res["cam_lim"] = [int(box[0] * sw), int((box[0] + box[2]) * sw)]
        res.update(_cam_params(sw, sh, cx, cy, fw, box))
        cam_faces = [[f for f in fs if abs(f[0] - cx) < 0.08 and abs(f[1] - cy) < 0.1] for fs in faces]
        res["cam_track"] = _track_camera(times, cam_faces, res["camcrop"][2] / sw) or [(0.0, cx)]
        # отдельное, более чуткое слежение для зоны вебки в «вебка + экран»: лицо держим ровно по центру
        res["split_track"] = _track_camera(times, cam_faces, res["cam"][2] / sw, dead_frac=0.07) or [(0.0, cx)]
        try:
            _people_on_cam(res, times, faces, box, cam_face, scam, sw, sh, cfg)
        except Exception as e:  # не должно ломать монтаж
            log.warning("Поиск второго человека на вебке не удался: %s", e)

    # --- лицо крупно (вебка на весь экран, IRL): кадр по лицу со слежением
    big_faces = [[f for f in fs if f[2] >= 0.06 and not (cam_face and abs(f[0] - cam_face[0]) < 0.05
                                                          and abs(f[1] - cam_face[1]) < 0.07)] for fs in faces]
    if any(big_faces):
        fws = [max(fs, key=lambda f: f[2])[2] for fs in big_faces if fs]
        cys = [max(fs, key=lambda f: f[2])[1] for fs in big_faces if fs]
        res.update(_crop_params(sw, sh, float(np.median(fws)), float(np.median(cys)), times, big_faces))
        try:
            _people_full(res, times, big_faces, sw, sh, cfg)
        except Exception as e:
            log.warning("Поиск нескольких людей в кадре не удался: %s", e)

    # --- режим для каждого кадра
    if cam_rel:
        cx, cy, _ = cam_face
        on_cam = [any(abs(f[0] - cx) < 0.06 and abs(f[1] - cy) < 0.09 for f in fs) for fs in faces]
        scene_full_cam = n and sum(on_cam) < 0.3 * n and sum(1 for bf in big_faces if bf and
                                                              max(f[2] for f in bf) >= 0.11) >= 0.6 * n
        if scene_full_cam and res.get("crop"):
            # стример переключил сцену на вебку во весь экран: маленькой вебки в кадре нет, есть его крупное лицо
            full = res.get("full_mode") or "crop"
            modes = [full if bf else "split" for bf in big_faces]
            res["screen_active"] = 0.0  # экрана в кадре нет — только камера
        else:
            modes, act = _cam_modes(times, frames, faces, cam_rel, cam_face, words, dur, cfg, hint)
            res.update(act)
        gw = int(min(sw, sh * W / (H - res["top_h"]))) // 2 * 2
        try:
            res["screen_x"] = _screen_focus(_screen_motion(frames, cam_rel), sw, gw)
        except Exception:
            pass
    else:
        full = res.get("full_mode") or "crop"
        if full != "crop":
            # несколько человек весь клип: держим общую раскладку и там, где детектор на кадре никого не нашёл
            modes = [full] * n
        else:
            modes = ["crop" if bf else "blur" for bf in big_faces]
    res["modes"] = modes
    if want != "auto":
        if want in ("split", "cam") and not cam_rel:
            want = "crop" if res.get("crop") else "blur"
        if want == "crop" and not res.get("crop"):
            want = "cam" if cam_rel else "blur"
        if want == "crop" and res.get("full_mode") in ("stack", "group"):
            want = res["full_mode"]  # в кадре несколько человек — показываем всех
        res["layout"] = want
        res["segments"] = [{"t0": 0.0, "t1": dur, "layout": want}]
        return res
    segs = _smooth_modes(times, modes, dur, min_len=float(r.get("layout_min_seconds", 5.0))) if modes \
        else [{"t0": 0.0, "t1": dur, "layout": "blur"}]
    if not r.get("dynamic_layout", True) and segs:
        main = max(segs, key=lambda x: x["t1"] - x["t0"])["layout"]
        segs = [{"t0": 0.0, "t1": dur, "layout": main}]
    res["segments"] = segs
    res["layout"] = max(segs, key=lambda x: x["t1"] - x["t0"])["layout"]
    return res


def _people_on_cam(res: dict, times: list, faces: list, box: list[float], cam_face: tuple, scam: dict | None,
                   sw: int, sh: int, cfg: dict) -> None:
    """На вебке не один человек (соведущий сидит рядом) или у второго участника своя вебка —
    показываем всех: в «вебка + экран» зона вебки берёт всех людей; в «только вебка» — каждый в своей
    полосе (stack) или общий кадр всех (group_rect)."""
    from . import people as pp
    r = cfg["render"]
    if not r.get("show_all_people", True) or not times:
        return
    W, H = int(r["width"]), int(r["height"])
    fw = cam_face[2]
    grp = [p for p in pp.find_people(times, faces, region=box, min_share=0.2) if 0.5 < p["w"] / max(fw, 1e-6) < 2.0]
    bounds: list = [box] * len(grp)
    for o in (scam or {}).get("others") or []:
        ox, oy, ofw = o["face"]
        ob = o.get("box") or _default_cam_box(ox, oy, ofw, sw, sh)
        mine = pp.find_people(times, faces, region=ob, min_share=0.2)
        if mine:
            grp.append(max(mine, key=lambda p: p["share"]))
            bounds.append(ob)
    if len(grp) < 2:
        return
    order = sorted(range(len(grp)), key=lambda i: (bounds[i][1] + bounds[i][3] / 2 > 0.5, grp[i]["cx"]))
    grp, bounds = [grp[i] for i in order], [bounds[i] for i in order]
    res["people"] = len(grp)
    max_up = float(r.get("cam_max_upscale", 4.5))
    if len(grp) <= 3 and min(p["share"] for p in grp) >= 0.35:
        n = len(grp)
        pan = pp.panels(grp, times, sw, sh, W, H // n // 2 * 2, bounds, max_up, _track_camera)
        if pan:
            res["stack"] = pan
    same_box = all(b == box for b in bounds)
    if not same_box and not res.get("stack") and len(grp) <= 3:
        # у каждого своя вебка, общего кадра нет — полосы, пусть и чуть мягче по резкости
        n = len(grp)
        pan = pp.panels(grp, times, sw, sh, W, H // n // 2 * 2, bounds, max_up * 1.6, _track_camera)
        if pan:
            res["stack"] = pan
    if same_box:
        # «вебка + экран»: зона вебки — все люди целиком (по пропорциям самой вебки)
        top = top_height(box, sw, sh, cfg)
        res["top_h"] = top
        res["top_rect"] = pp.expand_to_aspect(pp.group_rect(grp, sw, sh, box), W / top, sw, sh, box)
        res["group_rect"] = pp.expand_to_aspect(pp.group_rect(grp, sw, sh, box), W / (H * 0.42), sw, sh, box)
    else:
        top = int(res.get("top_h") or cam_height(cfg))
        n = len(grp)
        pan = pp.panels(grp, times, sw, sh, W // n // 2 * 2, top, bounds, max_up * 1.3, _track_camera)
        if pan:
            res["top_panels"] = pan
    log.info("На вебке/вебках %d человека: %s", len(grp),
             "по полосам" if res.get("stack") else "общим кадром" if res.get("group_rect") else "по вебкам")


def _people_full(res: dict, times: list, big_faces: list, sw: int, sh: int, cfg: dict) -> None:
    """Камера на весь кадр (IRL, подкаст, сцена «только вебка»), в кадре несколько человек."""
    from . import people as pp
    r = cfg["render"]
    if not r.get("show_all_people", True) or not times or not res.get("crop"):
        return
    W, H = int(r["width"]), int(r["height"])
    grp = pp.find_people(times, big_faces, min_share=0.25)
    if len(grp) < 2:
        return
    res["people"] = len(grp)
    cw = res["crop"][0]
    span = (max(p["cx"] + 0.8 * p["w"] for p in grp) - min(p["cx"] - 0.8 * p["w"] for p in grp)) * sw
    if span <= 0.92 * cw:
        # все и так помещаются в вертикальный кадр — камера держит центр группы
        mid = (max(p["cx"] for p in grp) + min(p["cx"] for p in grp)) / 2
        res["track"] = [(0.0, round(mid, 4))]
        return
    max_up = float(r.get("cam_max_upscale", 4.5))
    if len(grp) <= 3:
        n = len(grp)
        pan = pp.panels(grp, times, sw, sh, W, H // n // 2 * 2, [None] * n, max_up, _track_camera)
        if pan:
            res["stack"] = pan
            res["full_mode"] = "stack"
            log.info("В кадре %d человека — каждый в своей полосе", n)
            return
    res["group_rect"] = pp.expand_to_aspect(pp.group_rect(grp, sw, sh), W / (H * 0.5), sw, sh)
    res["full_mode"] = "group"
    log.info("В кадре %d человек — общий кадр", len(grp))


def _split_cam_crop(box: list[float], face: tuple, sw: int, sh: int, aspect: float, zoom: float = 1.0) -> list[float]:
    """Кадр вебки для «вебка + экран»: лицо по центру по горизонтали и чуть выше центра по вертикали,
    кадр не выходит за картинку вебки. Самый крупный план, при котором лицо остаётся по центру;
    zoom > 1 — ещё крупнее."""
    bx, by, bw, bh = box[0] * sw, box[1] * sh, box[2] * sw, box[3] * sh
    fx, fy, fwp = face[0] * sw, face[1] * sh, face[2] * sw
    fx = float(np.clip(fx, bx + 1, bx + bw - 1))
    fy = float(np.clip(fy, by + 1, by + bh - 1))
    hw = min(fx - bx, bx + bw - fx)              # сколько места слева/справа от лица внутри вебки
    ch = min(bh, (fy - by) / 0.42, (by + bh - fy) / 0.58, 2 * hw / aspect)
    ch = max(ch, min(bh, bw / aspect, 2.6 * fwp))  # не приближаем сверх меры, если лицо у самого края
    ch = min(ch / max(1.0, zoom), bh, bw / aspect)
    cw = ch * aspect
    x = float(np.clip(fx - cw / 2, bx, bx + bw - cw))
    y = float(np.clip(fy - ch * 0.42, by, by + bh - ch))
    return [x / sw, y / sh, cw / sw, ch / sh]


def _fit_aspect(box: list[float], face: tuple, sw: int, sh: int, aspect: float) -> list[float]:
    """Наибольший кадр с нужным соотношением сторон ВНУТРИ рамки вебки, лицо — ближе к центру."""
    bx, by, bw, bh = box[0] * sw, box[1] * sh, box[2] * sw, box[3] * sh
    fx, fy = face[0] * sw, face[1] * sh
    if bw / max(1.0, bh) > aspect:
        ch, cw = bh, bh * aspect
    else:
        cw, ch = bw, bw / aspect
    x = float(np.clip(fx - cw / 2, bx, bx + bw - cw))
    y = float(np.clip(fy - ch * 0.42, by, by + bh - ch))
    return [x / sw, y / sh, cw / sw, ch / sh]


def cam_height(cfg: dict) -> int:
    return int(cfg["render"].get("cam_height", 820)) // 2 * 2


def _cam_params(sw: int, sh: int, cx: float, cy: float, fw: float, box: list[float] | None = None) -> dict:
    """Только вебка на весь экран: кадр 9:16 вокруг лица, лицо крупно, не выходя за картинку вебки."""
    bx, by, bw, bh = (box or [0, 0, 1, 1])
    bx, by, bw, bh = bx * sw, by * sh, bw * sw, bh * sh
    face_px = fw * sw
    ch = min(face_px * 5.0, bh, sh)
    cw = min(ch * 9 / 16, bw)
    ch = min(ch, cw * 16 / 9)
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    x0 = int(np.clip(cx * sw - cw / 2, bx, max(bx, bx + bw - cw)))
    y0 = int(np.clip(cy * sh - ch * 0.40, by, max(by, by + bh - ch)))
    return {"camcrop": [max(0, x0), max(0, y0), cw, ch], "cam_bounds": [int(bx), int(bw)]}


def _crop_params(sw: int, sh: int, fw: float, cy: float, times: list, faces: list, static_cx: float | None = None) -> dict:
    """Вертикальное кадрирование 9:16 по лицу со слежением. Если лицо мелкое (общий план IRL) — приближаем."""
    zoom = 1.0 if fw >= 0.07 else 0.82
    ch = int(sh * zoom) // 2 * 2
    cw = int(min(sw, ch * 9 / 16)) // 2 * 2
    y0 = int(np.clip(cy * sh - ch * 0.42, 0, sh - ch)) if zoom < 1 else 0
    keys = _track_camera(times, faces, cw / sw) or [(0.0, static_cx if static_cx is not None else 0.5)]
    return {"crop": [cw, ch, y0], "track": [(round(t, 2), round(x, 4)) for t, x in keys]}


def _crop_x_expr(keys: list, sw: int, cw: int, lo: float = 0.0, hi: float | None = None, t_shift: float = 0.0) -> str:
    """Выражение ffmpeg для x(t): плавное движение камеры между опорными точками (в пределах [lo, hi])."""
    hi = (sw - cw) if hi is None else hi

    def px(x):
        return float(np.clip(x * sw - cw / 2, lo, max(lo, hi)))
    keys = [(t - t_shift, x) for t, x in keys]
    if len(keys) == 1:
        return f"{px(keys[0][1]):.1f}"
    expr = f"{px(keys[-1][1]):.1f}"
    for i in range(len(keys) - 1, 0, -1):
        t1, x1 = keys[i]
        x0 = px(keys[i - 1][1])
        x1p = px(x1)
        t0 = max(keys[i - 1][0], t1 - 0.8)
        seg = f"{x0:.1f}+({x1p - x0:.1f})*(t-{t0:.2f})/{max(0.05, t1 - t0):.2f}"
        expr = f"if(lt(t,{t0:.2f}),{x0:.1f},if(lt(t,{t1:.2f}),{seg},{expr}))"
    return expr


def _overlap(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def focus_rect(focus: list[float] | None, sw: int, sh: int, aspect: float, max_grow: float = 1.6,
               min_w: float = 0.24, min_h: float = 0.2, avoid: list[float] | None = None) -> tuple[int, int, int, int]:
    """Область экрана для показа крупно: прямоугольник «важного» (чат, донат, видео, игра) расширяется
    до нужных пропорций, но не больше чем в max_grow раз по каждой стороне (лишнее — размытым фоном),
    и не меньше min_w/min_h кадра (иначе картинка рассыплется на пиксели). avoid — вебка: расширяемся так,
    чтобы не захватить её больше, чем исходная область. Возвращает x, y, w, h в пикселях."""
    x, y, w, h = focus or [0.0, 0.0, 1.0, 1.0]
    cx, cy = (x + w / 2) * sw, (y + h / 2) * sh
    w0, h0 = max(w, min_w) * sw, max(h, min_h) * sh
    orig = (float(np.clip(cx - w0 / 2, 0, sw - w0)), float(np.clip(cy - h0 / 2, 0, sh - h0)), w0, h0)
    av = (avoid[0] * sw, avoid[1] * sh, avoid[2] * sw, avoid[3] * sh) if avoid else None
    base = _overlap(orig, av) if av else 0.0

    def build(g: float):
        w, h = w0, h0
        if w / h < aspect:
            w = min(h * aspect, w * g)
        else:
            h = min(w / aspect, h * g)
        w, h = min(w, sw), min(h, sh)
        # среди положений, которые накрывают исходную область, — с наименьшим захватом вебки
        xs = np.linspace(max(0, orig[0] + orig[2] - w), min(sw - w, orig[0]), 7) if w >= orig[2] else [orig[0]]
        ys = np.linspace(max(0, orig[1] + orig[3] - h), min(sh - h, orig[1]), 7) if h >= orig[3] else [orig[1]]
        best = None
        for x0 in xs:
            for y0 in ys:
                r = (float(x0), float(y0), w, h)
                ov = _overlap(r, av) if av else 0.0
                d = abs(x0 + w / 2 - cx) + abs(y0 + h / 2 - cy)
                if best is None or (ov, d) < best[0]:
                    best = ((ov, d), r)
        return best[1], best[0][0]
    g = max_grow
    rect, ov = build(g)
    while av and g > 1.0 and ov > base + 0.02 * sw * sh:
        g = max(1.0, g - 0.1)
        rect, ov = build(g)
    x0, y0, w, h = rect
    return int(x0) // 2 * 2, int(y0) // 2 * 2, int(w) // 2 * 2, int(h) // 2 * 2


def fill_rect(focus: list[float] | None, sw: int, sh: int, aspect: float, avoid: list[float] | None = None,
              min_w: float = 0.2) -> tuple[int, int, int, int]:
    """«Приблизить максимально»: кадр ровно нужных пропорций внутри важной области (игра, видео) —
    лишнее по краям обрезается, без размытых полос. Центр — по центру области; если рядом наложена
    вебка стримера, кадр сдвигается так, чтобы её не захватить."""
    x, y, w, h = (focus or [0.0, 0.0, 1.0, 1.0])[:4]
    fx, fy, fw, fh = x * sw, y * sh, w * sw, h * sh
    if fw / max(1.0, fh) > aspect:
        cw, ch = fh * aspect, fh
    else:
        cw, ch = fw, fw / aspect
    if cw < min_w * sw:  # слишком сильное увеличение — «мыло»
        k = min_w * sw / cw
        cw, ch = cw * k, ch * k
    if ch > sh:
        cw, ch = sh * aspect, sh
    if cw > sw:
        cw, ch = sw, sw / aspect
    cx, cy = fx + fw / 2, fy + fh / 2
    av = (avoid[0] * sw, avoid[1] * sh, avoid[2] * sw, avoid[3] * sh) if avoid else None
    xs = np.linspace(max(0.0, min(fx, sw - cw)), max(0.0, min(fx + fw - cw, sw - cw)), 9) if cw < fw else \
        [float(np.clip(cx - cw / 2, 0, sw - cw))]
    ys = np.linspace(max(0.0, min(fy, sh - ch)), max(0.0, min(fy + fh - ch, sh - ch)), 9) if ch < fh else \
        [float(np.clip(cy - ch / 2, 0, sh - ch))]
    best = None
    for x0 in xs:
        for y0 in ys:
            r = (float(x0), float(y0), cw, ch)
            ov = _overlap(r, av) / (cw * ch) if av else 0.0
            d = abs(x0 + cw / 2 - cx) / sw + abs(y0 + ch / 2 - cy) / sh
            key = (round(ov, 2), d)
            if best is None or key < best[0]:
                best = (key, r)
    x0, y0, cw, ch = best[1]
    return int(x0) // 2 * 2, int(y0) // 2 * 2, int(cw) // 2 * 2, int(ch) // 2 * 2


def _fit_panel(src_label: str, rect: tuple, PW: int, PH: int, tag: str, out: str) -> str:
    """Показать область rect в панели PW×PH целиком (без обрезки): по центру, а свободное место —
    размытой копией той же области."""
    x, y, w, h = rect
    if abs(w / h - PW / PH) < 0.04:  # пропорции совпали — просто масштаб
        return f"{src_label}crop={w}:{h}:{x}:{y},scale={PW}:{PH}:flags=lanczos{out}"
    return (f"{src_label}crop={w}:{h}:{x}:{y},split=2[{tag}fa][{tag}fb];"
            f"[{tag}fa]scale={PW // 4}:{PH // 4}:force_original_aspect_ratio=increase,crop={PW // 4}:{PH // 4},"
            f"boxblur=10:2,eq=brightness=-0.12,scale={PW}:{PH}[{tag}fg];"
            f"[{tag}fb]scale={PW}:{PH}:force_original_aspect_ratio=decrease:flags=lanczos[{tag}ff];"
            f"[{tag}fg][{tag}ff]overlay=(W-w)/2:(H-h)/2{out}")


def _panels_chain(inp: str, panels: list[dict], PW: int, PH: int, horizontal: bool, sw: int, tag: str, out: str,
                  t_shift: float = 0.0, total: int | None = None) -> str:
    """Несколько людей — каждый в своей полосе (друг под другом или рядом). У каждой полосы своя
    камера, которая следит за своим человеком. total — полный размер по оси склейки (остаток от деления
    отдаётся последней полосе)."""
    n = len(panels)
    sizes = [PH if not horizontal else PW] * n
    if total:
        sizes[-1] = total - sum(sizes[:-1])
    parts = [f"{inp}split={n}" + "".join(f"[{tag}i{k}]" for k in range(n))]
    for k, p in enumerate(panels):
        w_, h_ = (PW, sizes[k]) if not horizontal else (sizes[k], PH)
        xexpr = _crop_x_expr(p.get("track") or [(0.0, (p["lo"] + p["w"] / 2) / sw)], sw, p["w"], lo=p["lo"],
                             hi=max(p["lo"], p["hi"]), t_shift=t_shift)
        parts.append(f"[{tag}i{k}]crop=w={p['w']}:h={p['h']}:x='{xexpr}':y={p['y']},"
                     f"scale={w_}:{h_}:force_original_aspect_ratio=increase:flags=lanczos,crop={w_}:{h_}[{tag}q{k}]")
    stack = "hstack" if horizontal else "vstack"
    parts.append("".join(f"[{tag}q{k}]" for k in range(n)) + f"{stack}=inputs={n}{out}")
    return ";".join(parts)


def _layout_chain(layout: str, li: dict, cfg: dict, inp: str, out: str, tag: str, t_shift: float = 0.0,
                  focus: list[float] | None = None) -> str:
    """Цепочка фильтров одной раскладки: inp → кадр 1080×1920 → out.
    focus — важная область экрана [x, y, w, h] в долях кадра (от ИИ-режиссёра): её показываем крупно."""
    r = cfg["render"]
    W, H = int(r["width"]), int(r["height"])
    sw, sh = li["src_w"], li["src_h"]
    # --- несколько человек: каждый в своей полосе / общий кадр всех
    if layout in ("stack", "cam") and li.get("stack"):
        n = len(li["stack"])
        return _panels_chain(inp, li["stack"], W, H // n // 2 * 2, False, sw, tag, out, t_shift, total=H)
    if layout in ("group", "cam") and li.get("group_rect"):
        return _fit_panel(inp, tuple(li["group_rect"]), W, H, tag, out)
    if layout in ("stack", "group"):
        layout = "crop"
    if layout == "screen":
        if focus is not None and len(focus) > 4 and focus[4]:  # показать область целиком
            rect = focus_rect(focus[:4], sw, sh, W / H, max_grow=1.8, min_w=0.3, min_h=0.3, avoid=li.get("cam_box"))
        else:
            rect = fill_rect(focus, sw, sh, W / H, avoid=li.get("cam_box"), min_w=0.3)
        return _fit_panel(inp, rect, W, H, tag, out)
    if layout == "split" and li.get("cam"):
        top = int(li.get("top_h") or cam_height(cfg))
        x, y, cw, ch = li["cam"]
        bot_h = H - top
        xc = str(x)
        lim = li.get("cam_lim")
        track = li.get("split_track") or li.get("cam_track")
        if lim and lim[1] - lim[0] > cw + 8 and track:
            xc = "'" + _crop_x_expr(track, sw, cw, lo=lim[0], hi=lim[1] - cw, t_shift=t_shift) + "'"
        if li.get("top_rect"):        # на вебке несколько человек — в зоне вебки все
            top_chain = _fit_panel(f"[{tag}a]", tuple(li["top_rect"]), W, top, tag + "u", f"[{tag}t]")
        elif li.get("top_panels"):    # у участников разные вебки — рядом друг с другом
            n = len(li["top_panels"])
            top_chain = _panels_chain(f"[{tag}a]", li["top_panels"], W // n // 2 * 2, top, True, sw, tag + "u",
                                      f"[{tag}t]", t_shift, total=W)
        else:
            top_chain = (f"[{tag}a]crop={cw}:{ch}:{xc}:{y},scale={W}:{top}:force_original_aspect_ratio=increase:"
                         f"flags=lanczos,crop={W}:{top}[{tag}t]")
        if focus is not None:
            if len(focus) > 4 and focus[4]:  # текст доната, чат, переписка — целиком, без обрезки краёв
                rect = focus_rect(focus[:4], sw, sh, W / bot_h, avoid=li.get("cam_box"))
            else:                              # игра, видео — приблизить максимально
                rect = fill_rect(focus, sw, sh, W / bot_h, avoid=li.get("cam_box"))
            bottom = _fit_panel(f"[{tag}b]", rect, W, bot_h, tag, f"[{tag}d]")
        else:
            gw = int(min(sw, sh * W / bot_h)) // 2 * 2
            gx = int(np.clip(li.get("screen_x", (sw - gw) // 2), 0, sw - gw))
            bottom = (f"[{tag}b]crop={gw}:{sh}:{gx}:0,scale={W}:{bot_h}:force_original_aspect_ratio=increase:"
                      f"flags=lanczos,crop={W}:{bot_h}[{tag}d]")
        return (f"{inp}split=2[{tag}a][{tag}b];" + top_chain + ";" + bottom + ";"
                f"[{tag}t][{tag}d]vstack{out}")
    if layout == "cam" and li.get("camcrop") and li.get("cam_box") and \
            H / max(1, li["camcrop"][3]) > float(r.get("cam_max_upscale", 4.5)):
        # вебка маленькая: если растянуть её лицо на весь экран — будет «мыло». Показываем вебку целиком
        # (в ширину кадра), а сверху и снизу — её же размытую копию
        up = float(r.get("cam_max_upscale", 4.5))
        bx, by, bw, bh = li["cam_box"]
        ph = min(H, int(bh * sh * up))           # высота «окна» с лицом при допустимом увеличении
        fx = li.get("face_cx", (bx + bw / 2) * sw) / sw
        fy = li.get("face_cy", (by + bh / 2) * sh) / sh
        x, y, w, h = _fit_aspect(li["cam_box"], (fx, fy), sw, sh, W / ph)
        rect = (int(x * sw) // 2 * 2, int(y * sh) // 2 * 2, int(w * sw) // 2 * 2, int(h * sh) // 2 * 2)
        return _fit_panel(inp, rect, W, H, tag, out)
    if layout == "cam" and li.get("camcrop"):
        x0, y0, cw, ch = li["camcrop"]
        bx, bw = li.get("cam_bounds", [0, sw])
        xexpr = _crop_x_expr(li.get("cam_track") or [(0.0, (x0 + cw / 2) / sw)], sw, cw, lo=bx,
                             hi=bx + bw - cw, t_shift=t_shift)
        return (f"{inp}crop=w={cw}:h={ch}:x='{xexpr}':y={y0},scale={W}:{H}:flags=lanczos,"
                f"unsharp=5:5:0.6:5:5:0.0{out}")
    if layout == "crop" and li.get("crop"):
        cw, ch, y0 = li["crop"]
        xexpr = _crop_x_expr(li.get("track") or [(0.0, 0.5)], sw, cw, t_shift=t_shift)
        return f"{inp}crop=w={cw}:h={ch}:x='{xexpr}':y={y0},scale={W}:{H}:flags=lanczos{out}"
    return (f"{inp}split=2[{tag}a][{tag}b];"
            f"[{tag}a]scale={W//4}:{H//4}:force_original_aspect_ratio=increase,crop={W//4}:{H//4},"
            f"boxblur=12:2,eq=brightness=-0.10:saturation=1.2,scale={W}:{H}[{tag}g];"
            f"[{tag}b]scale={W}:-2:flags=lanczos[{tag}f];[{tag}g][{tag}f]overlay=(W-w)/2:(H-h)/2{out}")


def layout_chain(layout: str, li: dict, cfg: dict, inp: str, out: str, tag: str, t_shift: float = 0.0,
                 focus: list[float] | None = None) -> str:
    """Цепочка фильтров одной раскладки; квадратный пиксель на выходе, чтобы куски склеивались."""
    ch = _layout_chain(layout, li, cfg, inp, f"[{tag}o]", tag, t_shift, focus)
    return ch + f";[{tag}o]setsar=1{out}"


# ---------------------------------------------------------------- subtitles
def _ass_color(hex_rgb: str) -> str:
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def text_positions(layout: str, cfg: dict | None = None) -> tuple[int, int]:
    """(y субтитров, y хука) для раскладки. В «вебка + экран» субтитры — в верхней части экрана игры,
    чтобы не закрывать лицо и не попадать под интерфейс TikTok снизу."""
    if layout in ("split", "split_focus"):
        top = cam_height(cfg) if cfg else 820
        H = int(cfg["render"]["height"]) if cfg else 1920
        if layout == "split_focus":
            # внизу показываем что-то важное крупно (чат, донат, видео) — субтитры на стыке, чтобы не закрывать
            return top + 8, top
        return int(top + (H - top) * 0.5), top
    if layout.startswith("stack"):
        # несколько человек полосами: субтитры на стыке полос (на 2 полосах — по центру, на 3 — на нижнем стыке)
        H = int(cfg["render"]["height"]) if cfg else 1920
        n = int(layout[5:] or 2)
        return (H // 2 if n == 2 else H * (n - 1) // n), 330
    return {"crop": (1340, 330), "cam": (1400, 330), "blur": (1440, 430), "screen": (1500, 430),
            "group": (1500, 330)}.get(layout, (1440, 430))


def build_ass(words: list[dict], hook: str, layout, cfg: dict, prof: ProfanityFilter | None) -> str:
    """layout — строка или список сегментов [{t0, t1, layout}] в итоговой шкале."""
    r = cfg["render"]
    segs = layout if isinstance(layout, list) else [{"t0": 0.0, "t1": 1e9, "layout": layout}]

    def lay_at(t: float) -> str:
        for sg in segs:
            if t < sg["t1"]:
                return sg["layout"]
        return segs[-1]["layout"]
    sub_y, hook_y = text_positions(lay_at(0.0), cfg)
    hl = _ass_color(r["highlight_color"])
    font = r["font"]
    fs = int(r["font_size"])
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Sub,{font},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,6,3,5,70,70,0,1",
        f"Style: Hook,{font},{int(fs*0.8)},&H00111111,&H00111111,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,16,0,5,110,110,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hook and r.get("hook_title"):
        lines.append(f"Dialogue: 1,{_ass_time(0.05)},{_ass_time(r['hook_seconds'])},Hook,,0,0,0,,"
                     f"{{\\pos(540,{hook_y})\\fad(120,200)}}{_esc(hook)}")

    if r.get("subtitles") and words:
        groups: list[list[dict]] = []
        cur: list[dict] = []
        maxw = int(r.get("words_per_line") or 3)
        for w in words:
            if cur and (len(cur) >= maxw or w["s"] - cur[-1]["e"] > 0.6
                        or cur[-1]["w"].rstrip().endswith((".", "!", "?", ","))
                        or sum(len(x["w"]) for x in cur) + len(w["w"]) > 18):
                groups.append(cur)
                cur = []
            cur.append(w)
        if cur:
            groups.append(cur)

        def disp(w: dict) -> str:
            txt = w["w"].strip()
            if prof and cfg["censor"]["enabled"] and cfg["censor"]["subtitles"] and prof.is_profane(txt):
                txt = prof.mask(txt)
            return _esc(txt.upper().strip(",."))

        for gi, g in enumerate(groups):
            g_end = g[-1]["e"]
            nxt = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else None
            if nxt is not None and nxt - g_end < 0.35:
                g_end = nxt  # без мигания между близкими группами
            else:
                g_end += 0.25
            texts = [disp(w) for w in g]
            for wi, w in enumerate(g):
                st = w["s"] if wi else g[0]["s"]
                en = g[wi + 1]["s"] if wi + 1 < len(g) else g_end
                if en <= st:
                    continue
                parts = []
                for k, t in enumerate(texts):
                    parts.append(f"{{\\c{hl}&}}{t}{{\\c&H00FFFFFF&}}" if k == wi else t)
                pop = "\\fscx88\\fscy88\\t(0,90,\\fscx100\\fscy100)" if wi == 0 else ""
                gy = text_positions(lay_at(g[0]["s"]), cfg)[0]
                lines.append(f"Dialogue: 0,{_ass_time(st)},{_ass_time(en)},Sub,,0,0,0,,"
                             f"{{\\pos(540,{gy}){pop}}}{' '.join(parts)}")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- audio
def censor_intervals(words: list[dict], prof: ProfanityFilter, cfg: dict, dur: float) -> list[tuple[float, float]]:
    pad = cfg["censor"]["pad_ms"] / 1000
    iv = []
    for i in prof.find(words):
        s, e = max(0.0, words[i]["s"] - pad), min(dur, words[i]["e"] + pad)
        if e - s < 0.15:  # Whisper иногда даёт слишком короткие слова
            e = min(dur, s + 0.15)
        if iv and s <= iv[-1][1]:
            iv[-1] = (iv[-1][0], max(iv[-1][1], e))
        else:
            iv.append((s, e))
    return iv


def _gate(iv: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in iv) or "0"


def has_audio(path: Path) -> bool:
    p = run([ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
             "-of", "json", str(path)], check=False)
    try:
        return bool(json.loads(p.stdout.decode() or "{}").get("streams"))
    except ValueError:
        return False


# ------------------------------------------------------------------ encoder
_encoder_cache: dict = {}


def pick_encoder(cfg: dict) -> str:
    """Аппаратное кодирование, если доступно: NVIDIA (nvenc), AMD (amf), Intel (qsv), иначе процессор."""
    want = cfg["render"]["encoder"]
    if want != "auto":
        return want
    if "enc" not in _encoder_cache:
        _encoder_cache["enc"] = "libx264"
        for enc in ("h264_nvenc", "h264_amf", "h264_qsv"):
            p = run([ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=s=640x360:d=0.3", "-pix_fmt", "yuv420p",
                     "-c:v", enc, "-f", "null", "-"], check=False)
            if p.returncode == 0:
                _encoder_cache["enc"] = enc
                break
        log.info("Кодировщик видео: %s", _encoder_cache["enc"])
    return _encoder_cache["enc"]


# Битрейт ограничен (~12 Мбит/с): качество для TikTok то же, а видео плавно играет даже на слабом
# ноутбуке/по Wi-Fi. Ключевой кадр раз в 2 с — быстрая перемотка.
VCODEC_ARGS = {
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "20", "-b:v", "8M",
                   "-maxrate", "12M", "-bufsize", "24M"],
    "h264_amf": ["-c:v", "h264_amf", "-quality", "quality", "-rc", "vbr_peak", "-b:v", "9M", "-maxrate", "12M",
                 "-bufsize", "24M"],
    "h264_qsv": ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", "20", "-maxrate", "12M",
                 "-bufsize", "24M"],
}
X264_EXTRA = ["-maxrate", "12M", "-bufsize", "24M"]


# ------------------------------------------------------------------- render
def zoom_expr(zooms: list[float], amount: float) -> str:
    """Огибающая «наезда»: быстрый вход 0.18 с, удержание ~1.1 с, мягкий выход 0.5 с."""
    parts = []
    for z in zooms:
        a, b, c, d = z - 0.18, z, z + 1.1, z + 1.6
        parts.append(f"clip(min((t-{a:.2f})/0.18,({d:.2f}-t)/0.5),0,1)")
    return f"(1+{amount:.3f}*min(1,{'+'.join(parts)}))" if parts else "1"


def out_fps(src: Path, cfg: dict) -> int:
    """Частота кадров клипа: 60 для 60-кадровых стримов (плавнее, без «рывков» при пересчёте 58→30), иначе 30."""
    want = cfg["render"].get("fps", "auto")
    if str(want) != "auto":
        return int(want)
    try:
        info = ffprobe_video(src)
        fr = float(info.get("fps") or 0)
    except Exception:
        fr = 0
    return 60 if fr > 45 else 30


class Timeline:
    """Соответствие времени исходного клипа и итогового (после вырезания пауз и ускорения)."""

    def __init__(self, keep: list[tuple[float, float]] | None, dur: float, speed: float = 1.0):
        self.keep = [(max(0.0, a), min(dur, b)) for a, b in (keep or [(0.0, dur)]) if b - a > 0.05] or [(0.0, dur)]
        self.speed = max(0.5, min(2.0, float(speed or 1.0)))  # 2.0 = x2
        self.cut_dur = sum(b - a for a, b in self.keep)
        self.final_dur = self.cut_dur / self.speed

    def cut(self, t: float) -> float:
        """Время на смонтированной (без пауз) шкале, до ускорения."""
        acc = 0.0
        for a, b in self.keep:
            if t < a:
                return acc
            if t <= b:
                return acc + (t - a)
            acc += b - a
        return acc

    def final(self, t: float) -> float:
        return self.cut(t) / self.speed

    @property
    def has_cuts(self) -> bool:
        return len(self.keep) > 1 or self.keep[0][0] > 0.01

    def select_expr(self) -> str:
        return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in self.keep)

    def setpts_expr(self) -> str:
        """Новые метки времени кадров по их НАСТОЯЩЕМУ времени (а не по номеру кадра): у записей Twitch
        частота плавает (например 56–58 к/с при заявленных 60), и счёт по номеру кадра давал рассинхрон
        со звуком и «рваное» видео."""
        parts = []
        a0 = self.keep[0][0]
        if a0 > 0.0005:
            parts.append(f"{a0:.4f}")
        for (pa, pb), (a, b) in zip(self.keep, self.keep[1:]):
            gap = a - pb
            if gap > 0.0005:
                parts.append(f"{gap:.4f}*gte(T,{a - 0.0005:.4f})")
        shift = "+".join(parts) or "0"
        return f"(T-({shift}))/TB"


def find_pauses(src: Path, offset: float, dur: float, words: list[dict], category: str, cfg: dict) -> list[tuple[float, float]]:
    """Умное вырезание пауз: убираем только «мёртвые» паузы в речи (стример задумался, тишина).
    Не трогаем паузы, где звучит смех, реакция или игра (громко), и короткие комедийные паузы перед панчлайном.
    Возвращает интервалы, которые ОСТАВИТЬ (время от начала клипа)."""
    import subprocess
    r = cfg["render"]
    if not words or len(words) < 3:
        return [(0.0, dur)]
    comedic = category in ("funny", "fail", "epic", "scare", "rage", "cringe", "chat", "wholesome")
    min_gap = float(r.get("pause_min_comedic", 1.2) if comedic else r.get("pause_min", 0.7))
    keep_pad = float(r.get("pause_keep", 0.15))
    sr, hop = 8000, 400  # 50 мс
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    p = subprocess.run([ffmpeg_bin(), "-v", "error", "-ss", f"{offset:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                        "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"], stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, **kwargs)
    x = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32)
    if len(x) < sr:
        return [(0.0, dur)]
    n = len(x) // hop
    rms = 20 * np.log10(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) / 32768 + 1e-9)

    def db(a, b):
        i0, i1 = int(a * sr / hop), max(int(a * sr / hop) + 1, int(b * sr / hop))
        return rms[i0:i1]

    speech = np.concatenate([db(w["s"], w["e"]) for w in words if w["e"] > w["s"]] or [rms])
    speech_lvl = float(np.median(speech))
    cuts: list[tuple[float, float]] = []
    # тишина до первого слова
    first = words[0]["s"]
    if first > 0.6:
        seg = db(0.05, first - 0.05)
        if len(seg) and np.percentile(seg, 80) < max(speech_lvl - 14, -55):
            cuts.append((0.0, first - keep_pad))
    for w0, w1 in zip(words, words[1:]):
        gap = w1["s"] - w0["e"]
        if gap < min_gap:
            continue
        seg = db(w0["e"] + 0.08, w1["s"] - 0.08)
        if not len(seg):
            continue
        # «мёртвая» тишина: заметно тише речи и нет всплесков (смех/крик/звук игры)
        if np.percentile(seg, 85) < speech_lvl - 12 or np.percentile(seg, 85) < -48:
            a, b = w0["e"] + keep_pad, w1["s"] - keep_pad
            if b - a >= 0.3:
                cuts.append((a, b))
    if not cuts:
        return [(0.0, dur)]
    keep, t = [], 0.0
    for a, b in cuts:
        if a > t:
            keep.append((t, a))
        t = b
    if t < dur:
        keep.append((t, dur))
    return keep


def render_clip(src: Path, offset: float, dur: float, words_rel: list[dict], hook: str,
                out: Path, cfg: dict, prof: ProfanityFilter | None, layout_info: dict | None = None,
                zooms: list[float] | None = None, keep: list[tuple[float, float]] | None = None,
                speed: float = 1.0, layout_hint: dict | None = None) -> dict:
    """Рендерит клип. words_rel — слова со временем от начала клипа (исходная шкала).
    keep — какие интервалы оставить (вырезание пауз), speed — ускорение (1.0–2.0)."""
    r = cfg["render"]
    W, H = int(r["width"]), int(r["height"])
    work = out.parent / (out.stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    tl = Timeline(keep, dur, speed)
    D = tl.final_dur

    fps = out_fps(src, cfg)
    li = dict(layout_info or detect_layout(src, offset, dur, cfg, words_rel, layout_hint))
    layout = li["layout"]
    if li.get("top_h"):
        # зона вебки подогнана под пропорции этой вебки — субтитры и склейки считаем от неё
        cfg = {**cfg, "render": {**cfg["render"], "cam_height": int(li["top_h"])}}
        r = cfg["render"]
    sw, sh = li["src_w"], li["src_h"]
    # движение камеры — в шкале после вырезания пауз
    for k in ("track", "cam_track", "split_track"):
        if li.get(k):
            li[k] = [(tl.cut(t), x) for t, x in li[k]]
    for k in ("stack", "top_panels"):
        if li.get(k):
            li[k] = [{**p, "track": [(tl.cut(t), x) for t, x in p.get("track") or []]} for p in li[k]]
    if layout == "cam" and not li.get("camcrop") and li.get("face_cx"):
        li.update(_cam_params(sw, sh, li["face_cx"] / sw, li.get("face_cy", sh // 2) / sh, li.get("face_w", 0.06)))

    # сегменты раскладки (может меняться внутри клипа) — в шкале после вырезания пауз
    segs = []
    for sg in (li.get("segments") or [{"t0": 0.0, "t1": dur, "layout": layout}]):
        a, b = tl.cut(sg["t0"]), tl.cut(sg["t1"])
        same = segs and segs[-1]["layout"] == sg["layout"] and segs[-1].get("focus") == sg.get("focus")
        if b - a < 0.5 and segs:
            segs[-1]["t1"] = b
            continue
        if same:
            segs[-1]["t1"] = b
        else:
            segs.append({"t0": a, "t1": b, "layout": sg["layout"], "focus": sg.get("focus")})
    segs = [s_ for s_ in segs if s_["t1"] - s_["t0"] > 0.05] or [{"t0": 0.0, "t1": tl.cut_dur, "layout": layout}]
    segs[0]["t0"], segs[-1]["t1"] = 0.0, tl.cut_dur + 1.0
    if len(segs) == 1:
        vf = layout_chain(segs[0]["layout"], li, cfg, "[vin]", "[v0]", "L0", focus=segs[0].get("focus"))
    else:
        n = len(segs)
        vf = f"[vin]split={n}" + "".join(f"[s{i}]" for i in range(n)) + ";"
        for i, sg in enumerate(segs):
            vf += (f"[s{i}]trim=start={sg['t0']:.3f}:end={sg['t1']:.3f},setpts=PTS-STARTPTS,"
                   + layout_chain(sg["layout"], li, cfg, "", f"[p{i}]", f"L{i}", t_shift=sg["t0"],
                                  focus=sg.get("focus")) + ";")
        vf += "".join(f"[p{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v0]"
    def sub_layout(sg: dict) -> str:
        lay = sg["layout"]
        if lay in ("stack", "cam") and li.get("stack"):
            return f"stack{len(li['stack'])}"
        if lay in ("group", "cam") and li.get("group_rect"):
            return "group"
        return lay + ("_focus" if sg.get("focus") and lay == "split" else "")
    segs_final = [{"t0": sg["t0"] / tl.speed, "t1": sg["t1"] / tl.speed, "layout": sub_layout(sg)} for sg in segs]

    # субтитры и заглушение — в итоговой шкале
    words_final = [{**w, "s": tl.final(w["s"]), "e": tl.final(w["e"])} for w in words_rel
                   if tl.final(w["e"]) - tl.final(w["s"]) > 0.01]
    (work / "subs.ass").write_text(build_ass(words_final, hook, segs_final, cfg, prof), encoding="utf-8")

    pre = "[0:v]"
    if tl.has_cuts:
        pre += f"select='{tl.select_expr()}',setpts='{tl.setpts_expr()}',"
    vf = f"{pre}null[vin];" + vf

    ass_opt = "ass=subs.ass"
    fonts = ROOT_DIR / "fonts"
    if fonts.is_dir() and any(fonts.iterdir()):
        rel = os.path.relpath(fonts, work).replace("\\", "/")
        ass_opt += f":fontsdir={rel}"
    post = (f"setpts=PTS/{tl.speed:.4f}," if tl.speed != 1.0 else "") + f"fps={fps}"
    zooms_f = [tl.final(z) for z in (zooms or [])] if r.get("zoom", True) else []
    zooms_f = [z for z in zooms_f if 0.8 < z < D - 1.2]
    if zooms_f:
        # «наезд» камеры на реакцию: картинка плавно увеличивается и возвращается
        zexpr = zoom_expr(zooms_f, float(r.get("zoom_amount", 0.12)))
        post += (f",scale=w='trunc({W}*{zexpr}/2)*2':h='trunc({H}*{zexpr}/2)*2':eval=frame:flags=bicubic,"
                 f"crop={W}:{H}")
    post += f",{ass_opt}"
    if r.get("fade_out", True) and D > 3:
        post += f",fade=t=out:st={D - 0.35:.2f}:d=0.35"
    vf += f";[v0]{post},format=yuv420p[vout]"

    audio = has_audio(src)
    censored: list = []
    if audio:
        af = "[0:a]aresample=48000,aformat=channel_layouts=stereo"
        if tl.has_cuts:
            # без щелчков на стыках: короткое затухание у каждой границы вырезанного куска
            bounds = sorted({round(b, 3) for a, b in tl.keep if b < dur - 0.01} | {round(a, 3) for a, b in tl.keep if a > 0.01})
            if bounds:
                env = "*".join(f"clip(abs(t-{c:.3f})/0.012,0,1)" for c in bounds)
                af += f",asetnsamples=n=256,volume='{env}':eval=frame"
            af += f",aselect='{tl.select_expr()}',asetpts=N/SR/TB"
        if r.get("loudnorm"):
            af += ",loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"
        if prof and cfg["censor"]["enabled"]:
            censored = censor_intervals(words_rel, prof, cfg, dur)
        cens_cut = [(tl.cut(s), tl.cut(e)) for s, e in censored]
        cens_cut = [(s, e) for s, e in cens_cut if e - s > 0.02]
        if cens_cut:
            mode = cfg["censor"].get("mode", "partial")
            ramp = 0.015
            if mode == "partial":
                # первые ~0.1 с слова звучат как есть — понятно, ЧТО сказано, остальное приглушено
                cens_cut = [(s + min(max(0.35 * (e - s), 0.07), 0.16), e) for s, e in cens_cut]
                cens_cut = [(s, e) for s, e in cens_cut if e - s > 0.04]
            env = "+".join(f"clip(min((t-{s - ramp:.3f})/{ramp},({e + ramp:.3f}-t)/{ramp}),0,1)" for s, e in cens_cut) or "0"
            env = f"min(1,{env})"
            if mode in ("partial", "muffle"):
                # «глухо, как через стену»: слово угадывается по ритму, но не разборчиво; без тишины
                f0 = int(cfg["censor"].get("muffle_freq", 320))
                g = float(cfg["censor"].get("muffle_gain", 0.55))
                af += (f",asetnsamples=n=256,asplit=2[cd][cw];"
                       f"[cw]lowpass=f={f0},lowpass=f={f0},lowpass=f={f0},volume={g},volume='{env}':eval=frame[cw2];"
                       f"[cd]volume='1-{env}':eval=frame[cd2];[cd2][cw2]amix=inputs=2:normalize=0")
            else:
                af += f",asetnsamples=n=256,volume='max(0,1-({env}))':eval=frame"
                if mode == "beep":
                    v, f = float(cfg["censor"]["beep_volume"]), float(cfg["censor"]["beep_freq"])
                    af += (f"[a0];aevalsrc='{v}*sin(2*PI*{f}*t)*({env})':s=48000:d={tl.cut_dur:.3f},"
                           f"lowpass=f=1800,aformat=channel_layouts=stereo[bp];"
                           f"[a0][bp]amix=inputs=2:duration=first:normalize=0")
        if tl.speed != 1.0:
            af += f",atempo={tl.speed:.4f}"
        af += f",afade=t=in:d=0.08,afade=t=out:st={max(0, D - 0.45):.2f}:d=0.45[aout]"
        graph = vf + ";" + af
    else:
        graph = vf

    enc = pick_encoder(cfg)
    vcodec = VCODEC_ARGS.get(enc) or ["-c:v", "libx264", "-preset", r["preset"], "-crf", str(r["crf"]),
                                      "-profile:v", "high", *X264_EXTRA]
    tmp_out = out.with_name(out.stem + ".rendering.mp4")
    hw = ["-hwaccel", "auto"] if r.get("hwdecode", True) else []
    cmd = [ffmpeg_bin(), "-y", "-v", "error", *hw, "-ss", f"{offset:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-filter_complex", graph, "-map", "[vout]"]
    if audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k"]
    cmd += vcodec + ["-g", str(fps * 2), "-r", str(fps), "-movflags", "+faststart", "-t", f"{D:.3f}", str(tmp_out.resolve())]
    try:
        run(cmd, cwd=work)
    except Exception:
        if enc == "libx264" and not hw:
            raise
        log.warning("Видеокарта не справилась (%s) — перекодирую на процессоре", enc)
        if hw:
            cmd = [c for c in cmd if c not in ("-hwaccel", "auto")]
        if enc != "libx264":
            _encoder_cache["enc"] = "libx264"
        i = cmd.index(vcodec[0])
        cmd[i:i + len(vcodec)] = ["-c:v", "libx264", "-preset", r["preset"], "-crf", str(r["crf"]), "-profile:v", "high",
                                  *X264_EXTRA]
        enc = "libx264"
        run(cmd, cwd=work)
    tmp_out.replace(out)

    thumb = out.with_suffix(".jpg")
    run([ffmpeg_bin(), "-y", "-v", "error", "-ss", f"{min(1.0, D / 2):.2f}", "-i", str(out.resolve()),
         "-frames:v", "1", "-vf", "scale=360:-2", str(thumb.resolve())], check=False)
    try:
        for p in work.iterdir():
            p.unlink()
        work.rmdir()
    except OSError:
        pass
    removed = round(dur - tl.cut_dur, 1)
    return {"layout": layout, "layouts": [sg["layout"] for sg in segs],
            "shots": [{"t0": round(sg["t0"] / tl.speed, 1), "layout": sg["layout"], "focus": sg.get("focus")} for sg in segs],
            "words_final": [[w["w"], round(w["s"], 2), round(w["e"], 2)] for w in words_final], "censored": [[round(s, 2), round(e, 2)] for s, e in censored], "encoder": enc,
            "zooms": [round(z, 1) for z in zooms_f], "src_quality": f"{sh}p", "speed": tl.speed,
            "pauses_removed": removed, "pause_cuts": len(tl.keep) - 1, "final_duration": round(D, 1)}
