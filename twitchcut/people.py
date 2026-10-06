"""Все участники разговора в кадре: соведущий рядом на вебке, гость, второй стример со своей вебкой,
несколько человек в IRL-кадре.

Лица из детектора (по кадрам клипа) связываются в «дорожки» — по одной на человека. Человек считается
участником, если он виден заметную часть клипа и его лицо не мелкое (прохожие и люди на фоне отсекаются).

Дальше раскладка решает, как показать всех:
  stack — каждый человек в своей горизонтальной полосе (2–3 полосы друг под другом), камера у каждой
          полосы своя и следит за своим человеком; так выглядят подкасты в TikTok;
  group — общий кадр со всеми людьми по центру, сверху и снизу — его же размытая копия (когда людей много
          или полосы получились бы слишком «мыльными»).
"""
from __future__ import annotations

import numpy as np


def _inside(f: tuple, region: list[float] | None, pad: float = 0.01) -> bool:
    if not region:
        return True
    x, y, w, h = region
    return x - pad <= f[0] <= x + w + pad and y - pad <= f[1] <= y + h + pad


def find_people(times: list[float], faces: list[list[tuple]], region: list[float] | None = None,
                min_share: float = 0.2, min_rel: float = 0.45, exclude: list[tuple] | None = None) -> list[dict]:
    """Дорожки людей. faces — по кадрам [(cx, cy, w, h, score)] в долях кадра. region — искать только внутри
    прямоугольника (рамка вебки). exclude — лица, которые не считаем (cx, cy, допуск).
    Возвращает людей слева направо: {cx, cy, w, h, share, pts: [(t, cx, cy, w, h)]}."""
    n = len(times)
    if not n:
        return []
    tracks: list[dict] = []
    for t, fs in zip(times, faces):
        fs = [f for f in fs if _inside(f, region)
              and not any(abs(f[0] - ex[0]) < ex[2] and abs(f[1] - ex[1]) < 1.5 * ex[2] for ex in (exclude or []))]
        pairs = sorted((abs(f[0] - tr["cx"]) + 0.5 * abs(f[1] - tr["cy"]), i, j)
                       for i, f in enumerate(fs) for j, tr in enumerate(tracks))
        used_f, used_t = set(), set()
        for d, i, j in pairs:
            if i in used_f or j in used_t:
                continue
            tr = tracks[j]
            if d > max(1.5 * tr["w"], 0.03):
                continue
            f = fs[i]
            tr["pts"].append((t, f[0], f[1], f[2], f[3]))
            tr["cx"], tr["cy"], tr["w"] = f[0], f[1], 0.7 * tr["w"] + 0.3 * f[2]
            used_f.add(i)
            used_t.add(j)
        for i, f in enumerate(fs):
            if i not in used_f:
                tracks.append({"cx": f[0], "cy": f[1], "w": f[2], "pts": [(t, f[0], f[1], f[2], f[3])]})
    # обрывки одного и того же человека (детектор на секунду потерял лицо и нашёл чуть в стороне) —
    # склеиваем дорожки, которые не пересекаются по времени и стоят рядом
    tracks.sort(key=lambda tr: -len(tr["pts"]))
    merged: list[dict] = []
    for tr in tracks:
        ts = {p[0] for p in tr["pts"]}
        mx, mw = float(np.median([p[1] for p in tr["pts"]])), float(np.median([p[3] for p in tr["pts"]]))
        for m in merged:
            if ts & {p[0] for p in m["pts"]}:
                continue
            if abs(mx - float(np.median([p[1] for p in m["pts"]]))) < 1.5 * max(mw, m["w"]):
                m["pts"] = sorted(m["pts"] + tr["pts"])
                break
        else:
            merged.append({"pts": list(tr["pts"]), "w": mw})
    need = max(2, int(np.ceil(min_share * n)))
    out = []
    for m in merged:
        if len({p[0] for p in m["pts"]}) < need:
            continue
        pts = m["pts"]
        out.append({"cx": float(np.median([p[1] for p in pts])), "cy": float(np.median([p[2] for p in pts])),
                    "w": float(np.median([p[3] for p in pts])), "h": float(np.median([p[4] for p in pts])),
                    "share": round(len({p[0] for p in pts}) / n, 2), "pts": pts})
    if not out:
        return []
    big = max(p["w"] for p in out)
    out = [p for p in out if p["w"] >= min_rel * big]
    out.sort(key=lambda p: p["cx"])
    return out


def per_frame(person: dict, times: list[float]) -> list[list[tuple]]:
    """Лица одного человека по кадрам (для слежения камерой)."""
    by_t = {}
    for t, cx, cy, w, h in person["pts"]:
        by_t[t] = [(cx, cy, w, h, 1.0)]
    return [by_t.get(t, []) for t in times]


def group_rect(people: list[dict], sw: int, sh: int, bounds: list[float] | None = None) -> list[int]:
    """Прямоугольник, в который входят все люди (голова и плечи), в пикселях, внутри bounds (доли кадра)."""
    bx, by, bw, bh = bounds or [0.0, 0.0, 1.0, 1.0]
    L, T, R, B = bx * sw, by * sh, (bx + bw) * sw, (by + bh) * sh
    x0 = max(L, min((p["cx"] - 1.0 * p["w"]) * sw for p in people))
    x1 = min(R, max((p["cx"] + 1.0 * p["w"]) * sw for p in people))
    y0 = max(T, min((p["cy"] - 1.1 * p["h"]) * sh for p in people))
    y1 = min(B, max((p["cy"] + 1.8 * p["h"]) * sh for p in people))
    return [int(x0) // 2 * 2, int(y0) // 2 * 2, int(x1 - x0) // 2 * 2, int(y1 - y0) // 2 * 2]


def expand_to_aspect(rect: list[int], aspect: float, sw: int, sh: int, bounds: list[float] | None = None) -> list[int]:
    """Расширяет прямоугольник до нужных пропорций, не выходя за bounds (если не хватает места —
    остаётся у́же/ниже, а недостающее потом закроет размытый фон)."""
    bx, by, bw, bh = bounds or [0.0, 0.0, 1.0, 1.0]
    L, T, R, B = bx * sw, by * sh, (bx + bw) * sw, (by + bh) * sh
    x, y, w, h = [float(v) for v in rect]
    cx, cy = x + w / 2, y + h / 2
    if w / max(1.0, h) < aspect:
        w = min(h * aspect, R - L)
    else:
        h = min(w / aspect, B - T)
    x = float(np.clip(cx - w / 2, L, R - w))
    y = float(np.clip(cy - h / 2, T, B - h))
    return [int(x) // 2 * 2, int(y) // 2 * 2, int(w) // 2 * 2, int(h) // 2 * 2]


def panels(people: list[dict], times: list[float], sw: int, sh: int, PW: int, PH: int,
           bounds: list[list[float] | None], max_up: float, track_fn) -> list[dict] | None:
    """Кадр для каждого человека (полоса PW×PH). bounds[i] — где можно кадрировать i-го (его вебка
    или весь кадр). None — если так показать нельзя (лицо слишком маленькое → «мыло», или соседи
    так близко, что полосы почти одинаковые)."""
    aspect = PW / PH
    # 1) для каждого человека — насколько крупно его МОЖНО показать (высота кадра в высотах лица)
    caps = []
    for i, p in enumerate(people):
        bx, by, bw, bh = bounds[i] or [0.0, 0.0, 1.0, 1.0]
        L, W_, H_ = bx * sw, bw * sw, bh * sh
        fh = p["h"] * sh
        cw = min(H_ * aspect, W_)
        # лицо должно быть примерно по центру полосы (не ближе 27 % к краю), даже если человек у края картинки
        cxp = p["cx"] * sw
        room = min(cxp - L, L + W_ - cxp)
        cw = min(cw, max(1.0, room / 0.27))
        # соседи в той же картинке: полоса не должна захватывать чужое лицо
        for j, q in enumerate(people):
            if j != i and bounds[j] == bounds[i]:
                cw = min(cw, 2 * (abs(q["cx"] - p["cx"]) * sw - 0.6 * q["w"] * sw))
        caps.append(cw / aspect / max(1.0, fh))
    # 2) один масштаб на всех — чтобы лица в полосах были одного размера
    k = min(min(caps), 3.2)
    if k < 1.5:
        return None
    out = []
    for i, p in enumerate(people):
        bx, by, bw, bh = bounds[i] or [0.0, 0.0, 1.0, 1.0]
        L, T, W_, H_ = bx * sw, by * sh, bw * sw, bh * sh
        ch = min(H_, k * p["h"] * sh)
        cw = min(W_, ch * aspect)
        ch = cw / aspect
        if PH / max(1.0, ch) > max_up:
            return None
        cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
        y = int(np.clip(p["cy"] * sh - ch * 0.5, T, T + H_ - ch)) // 2 * 2  # запас над головой
        keys = track_fn(times, per_frame(p, times), cw / sw) or [(0.0, p["cx"])]
        out.append({"w": cw, "h": ch, "y": y, "lo": int(L), "hi": int(L + W_ - cw), "track": keys})
    return out
