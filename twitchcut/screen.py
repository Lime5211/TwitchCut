"""Где на экране главное: окно видео/игры/сайта, на которое смотрит стример.

Окно ищется по самой картинке, а не по догадке ИИ (координаты от модели «съезжали» влево):
  - что меняется во времени (идёт видео, игра) — движение между соседними кадрами;
  - что отличается от «фона сцены» (страница браузера, рамки, логотипы, подложка OBS стоят на месте весь стрим,
    а окно с видео/статьёй/картинкой — каждый раз своё). Фон сцены — медиана кадров из многих клипов стрима,
    он копится и между стримами (data/scene_bg);
  - если ничего из этого нет — детализированная область на однотонной подложке (статья/картинка на сером фоне).
Найденный прямоугольник подтягивается к настоящим краям окна (устойчивые линии на кадрах).
Состояния экрана — отрезки времени с одним и тем же окном: когда открылось другое окно, это повод сменить кадр.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .util import ROOT_DIR, log

GW, GH = 320, 180


def small_gray(fr: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
    g = cv2.resize(g, (GW, GH), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (3, 3), 0)


# ------------------------------------------------------------------ фон сцены
def _bg_file(login: str, face: list) -> Path:
    key = f"{login}_{round(face[0] / 0.03)}_{round(face[1] / 0.03)}" if face else f"{login}_nocam"
    return ROOT_DIR / "data" / "scene_bg" / (hashlib.md5(key.encode()).hexdigest()[:10] + ".npz")


def load_bg(login: str, face: list | None) -> np.ndarray | None:
    """Фон сцены стримера (медиана накопленных кадров) или None."""
    p = _bg_file(login or "", face or [])
    if not p.exists():
        return None
    try:
        st = np.load(p)["stack"]
        if len(st) < 6:
            return None
        return np.median(st, axis=0).astype(np.uint8)
    except Exception:
        return None


def update_bg(login: str, face: list | None, frames: list, keep: int = 48) -> None:
    """Добавить кадры (из разных клипов, разных моментов) в накопитель фона сцены."""
    if not frames:
        return
    p = _bg_file(login or "", face or [])
    p.parent.mkdir(parents=True, exist_ok=True)
    new = np.stack([small_gray(f) for _, f in frames]).astype(np.uint8)
    try:
        old = np.load(p)["stack"] if p.exists() else np.zeros((0, GH, GW), np.uint8)
    except Exception:
        old = np.zeros((0, GH, GW), np.uint8)
    st = np.concatenate([old, new])
    if len(st) > keep:  # равномерно прореживаем, свежие кадры важнее
        idx = np.unique(np.concatenate([np.linspace(0, len(st) - len(new) - 1, max(1, keep - len(new) // 2)).astype(int),
                                        np.arange(len(st) - len(new), len(st))[-keep // 2:]]))
        st = st[idx][-keep:]
    try:
        np.savez_compressed(p, stack=st)
    except OSError as e:
        log.debug("Фон сцены не сохранён: %s", e)


# ------------------------------------------------------------- окно на экране
def _components(mask: np.ndarray, min_area: float = 0.05):
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < min_area * GW * GH or w < 0.12 * GW or h < 0.15 * GH:
            continue
        out.append((int(a), [x / GW, y / GH, w / GW, h / GH]))
    out.sort(key=lambda c: -c[0])
    return out


def _cam_mask(cam_box: list | None, pad: float = 0.015) -> np.ndarray:
    m = np.ones((GH, GW), bool)
    if cam_box:
        x, y, w, h = cam_box[:4]
        m[max(0, int((y - pad) * GH)):int((y + h + pad) * GH) + 1, max(0, int((x - pad) * GW)):int((x + w + pad) * GW) + 1] = False
    return m


def window_rects(frames: list, cam_box: list | None, bg: np.ndarray | None = None) -> list[list[float] | None]:
    """Для каждого кадра — прямоугольник главного окна [x, y, w, h] в долях кадра (или None)."""
    import cv2
    if not frames:
        return []
    G = [small_gray(f).astype(np.int16) for _, f in frames]
    keep = _cam_mask(cam_box)
    med = np.median(np.stack(G), axis=0) if len(G) >= 5 else None
    k_close = np.ones((7, 7), np.uint8)
    k_open = np.ones((5, 5), np.uint8)
    out: list = []
    for i, g in enumerate(G):
        mot = np.zeros((GH, GW), bool)
        for j in (i - 1, i + 1):
            if 0 <= j < len(G):
                mot |= np.abs(g - G[j]) > 18
        cands = []
        m = cv2.morphologyEx(cv2.morphologyEx((mot & keep).astype(np.uint8), cv2.MORPH_CLOSE, k_close),
                             cv2.MORPH_OPEN, k_open)
        cm = _components(m)
        if cm:
            cands.append(cm[0])
        if bg is not None:
            d = (np.abs(g - bg.astype(np.int16)) > 24) & keep
            m2 = cv2.morphologyEx(cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_CLOSE, k_close), cv2.MORPH_OPEN, k_open)
            c2 = _components(m2)
            if c2:
                cands.append(c2[0])
        elif med is not None and not cands:
            d = (np.abs(g - med) > 24) & keep
            m2 = cv2.morphologyEx(cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_CLOSE, k_close), cv2.MORPH_OPEN, k_open)
            c2 = _components(m2)
            if c2:
                cands.append(c2[0])
        if not cands:
            # статичная картинка/статья на однотонной подложке: детализированная область
            lap = np.abs(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F))
            tex = cv2.blur((lap > 12).astype(np.float32), (9, 9)) > 0.08
            m3 = cv2.morphologyEx(cv2.morphologyEx((tex & keep).astype(np.uint8), cv2.MORPH_CLOSE, k_close),
                                  cv2.MORPH_OPEN, k_open)
            c3 = _components(m3, 0.08)
            if c3:
                cands.append(c3[0])
        if not cands:
            out.append(None)
            continue
        # движение и отличие от фона вместе: окно — объединение, если они про одно и то же место
        a = max(cands, key=lambda c: c[0])[1]
        for _, b in cands:
            if b is not a and _iou(a, b) > 0.3:
                a = _union(a, b)
        out.append(a)
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def _union(a, b) -> list[float]:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0]


def screen_states(times: list[float], rects: list, frames: list, dur: float, cam_box: list | None = None,
                  min_len: float = 3.0) -> list[dict]:
    """Отрезки времени с одним и тем же главным окном: [{t0, t1, rect}]. Рамка окна в каждом отрезке —
    по медиане кадров и подтянута к настоящим краям окна."""
    groups: list[dict] = []
    for t, r, fr in zip(times, rects, frames):
        if r is None:
            continue
        if groups and _iou(groups[-1]["med"], r) > 0.55:
            g = groups[-1]
            g["rs"].append(r)
            g["fr"].append(fr)
            g["t1"] = t
            g["med"] = list(np.median(np.array(g["rs"]), axis=0))
        else:
            groups.append({"t0": t, "t1": t, "rs": [r], "fr": [fr], "med": list(r)})
    # дребезг: короткие группы, похожие на соседей, сливаем
    merged: list[dict] = []
    for g in groups:
        if merged and (g["t1"] - g["t0"] < min_len and len(g["rs"]) <= 2):
            merged[-1]["rs"] += g["rs"]
            merged[-1]["fr"] += g["fr"]
            merged[-1]["t1"] = g["t1"]
            continue
        merged.append(g)
    out = []
    for k, g in enumerate(merged):
        rect = [float(v) for v in np.median(np.array(g["rs"]), axis=0)]
        try:
            rect = snap_rect(rect, g["fr"] if len(g["fr"]) >= 3 else frames)
        except Exception:
            pass
        rect = _avoid_cam(rect, cam_box)
        t0 = 0.0 if k == 0 else (merged[k - 1]["t1"] + g["t0"]) / 2
        t1 = dur if k == len(merged) - 1 else (g["t1"] + merged[k + 1]["t0"]) / 2
        out.append({"t0": round(t0, 2), "t1": round(t1, 2), "rect": [round(v, 4) for v in rect]})
    return out


def _avoid_cam(rect: list[float], cam: list | None) -> list[float]:
    """Окно, которое заходит на вебку стримера краем, — обрезаем по вебке (вебка и так показана сверху)."""
    if not cam:
        return rect
    x, y, w, h = rect
    cx, cy, cw, ch = cam[:4]
    ix = min(x + w, cx + cw) - max(x, cx)
    iy = min(y + h, cy + ch) - max(y, cy)
    if ix <= 0 or iy <= 0 or ix * iy < 0.12 * w * h:
        return rect  # вебка лишь чуть заходит на угол окна — окно не режем (иначе пропадёт кусок видео)
    opts = []
    if cx > x + 0.5 * w:
        opts.append([x, y, cx - x, h])                  # вебка справа — отрезаем правую часть
    if cx + cw < x + 0.5 * w:
        opts.append([cx + cw, y, x + w - cx - cw, h])   # слева
    if cy > y + 0.5 * h:
        opts.append([x, y, w, cy - y])                  # снизу
    if cy + ch < y + 0.5 * h:
        opts.append([x, cy + ch, w, y + h - cy - ch])   # сверху
    opts = [o for o in opts if o[2] > 0.1 and o[3] > 0.1]
    if not opts:
        return rect
    return max(opts, key=lambda o: o[2] * o[3])


def snap_rect(rect: list[float], frames: list, reach: float = 0.035) -> list[float]:
    """Подтянуть стороны прямоугольника к устойчивым линиям кадра (край окна плеера/игры/сайта)."""
    from .render import _EdgeStack
    if len(frames) < 3:
        return rect
    es = _EdgeStack(frames)
    w, h = es.w, es.h
    L, T = rect[0] * w, rect[1] * h
    R, B = (rect[0] + rect[2]) * w, (rect[1] + rect[3]) * h
    dx, dy = int(reach * w), int(reach * h)

    def best_h(y0):
        cand = [(es.h_score(y, L, R), -abs(y - y0), y) for y in range(max(0, int(y0) - dy), min(h, int(y0) + dy + 1))]
        sc, _, y = max(cand) if cand else (0, 0, y0)
        return y if sc > 0.55 else y0

    def best_v(x0):
        cand = [(es.v_score(x, T, B), -abs(x - x0), x) for x in range(max(0, int(x0) - dx), min(w, int(x0) + dx + 1))]
        sc, _, x = max(cand) if cand else (0, 0, x0)
        return x if sc > 0.55 else x0
    T2, B2, L2, R2 = best_h(T), best_h(B), best_v(L), best_v(R)
    if R2 - L2 < 0.6 * (R - L) or B2 - T2 < 0.6 * (B - T):
        return rect
    return [L2 / w, T2 / h, (R2 - L2) / w, (B2 - T2) / h]


def state_at(states: list[dict], t: float) -> dict | None:
    for s in states:
        if s["t0"] <= t < s["t1"]:
            return s
    return states[-1] if states else None
