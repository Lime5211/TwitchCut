"""Проверка синхронности субтитров со звуком клипа.

Речь распознаётся по одному аудиофайлу (звук всего стрима или куска эфира), а клип монтируется из
отдельно скачанного видеофрагмента. Если эти два источника хоть немного разъехались, субтитры отстают
или вовсе не про то. Перед монтажом сравниваем «отпечатки» звука (огибающие по полосам частот) и:
- находим точный сдвиг и подвигаем слова субтитров;
- если звук вообще не совпадает — сигнал, что расшифровка не от этого места, и речь клипа нужно
  распознать заново прямо по его видеофрагменту.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from .util import ffmpeg_bin

SR = 16000
HOP = 400          # 25 мс
WIN = 1024
BANDS = np.geomspace(120, 5000, 21)
_NOHIDE = {"creationflags": 0x08000000} if os.name == "nt" else {}


def _pcm(path: Path, start: float, dur: float) -> np.ndarray:
    p = subprocess.run([ffmpeg_bin(), "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}",
                        "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **_NOHIDE)
    return np.frombuffer(p.stdout, dtype=np.float32)


def _features(x: np.ndarray) -> np.ndarray:
    n = (len(x) - WIN) // HOP
    if n <= 0:
        return np.zeros((0, len(BANDS) - 1), dtype=np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP][:n] * np.hanning(WIN).astype(np.float32)
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(WIN, 1 / SR)
    feats = np.stack([spec[:, (f >= BANDS[i]) & (f < BANDS[i + 1])].sum(1) for i in range(len(BANDS) - 1)], 1)
    feats = np.log(feats + 1e-7)
    feats -= feats.mean(1, keepdims=True)          # тембр, а не громкость
    feats = np.diff(feats, axis=0, prepend=feats[:1])  # изменения во времени — чёткий «отпечаток»
    return feats.astype(np.float32)


def measure_shift(ref: Path, ref_start: float, seg: Path, seg_start: float, dur: float,
                  search: float = 10.0) -> tuple[float, float]:
    """Где в ref (около ref_start) звучит то, что в seg начинается с seg_start.
    Возвращает (сдвиг в секундах: найдено − ожидалось, уверенность 0..1)."""
    dur = max(6.0, min(dur, 45.0))
    a = _features(_pcm(seg, seg_start, dur))
    lo = max(0.0, ref_start - search)
    b = _features(_pcm(ref, lo, dur + (ref_start - lo) + search))
    if len(a) < 40 or len(b) < len(a) + 2:
        return 0.0, 0.0
    a = (a - a.mean()) / (a.std() + 1e-6)
    n = len(a)
    best, best_i = -1.0, 0
    for i in range(len(b) - n + 1):
        w = b[i:i + n]
        sc = float(((w - w.mean()) / (w.std() + 1e-6) * a).mean())
        if sc > best:
            best, best_i = sc, i
    found = lo + best_i * HOP / SR
    return round(found - ref_start, 3), round(max(0.0, best), 3)


# ------------------------------------------------------------ тайминг слов
def _norm(w: str) -> str:
    import re
    t = re.sub(r"[^\w]+", "", w.lower().replace("ё", "е"))
    return t[:5]


def retime_words(orig: list[dict], fresh: list[dict], lo: float, hi: float) -> tuple[list[dict], dict]:
    """Переносит точные таймкоды из fresh (повторное распознавание звука клипа) на слова orig
    (их текст не трогаем). Совпадения ищутся по последовательности слов; слова без пары
    растягиваются между соседними совпавшими. Слова вне [lo, hi] не меняются.
    Возвращает (слова, отчёт)."""
    import difflib
    idx = [i for i, w in enumerate(orig) if lo <= (w["s"] + w["e"]) / 2 <= hi]
    rep = {"n": len(idx), "matched": 0, "shift": 0.0, "applied": False}
    if len(idx) < 3 or len(fresh) < 3:
        return orig, rep
    a = [_norm(orig[i]["w"]) for i in idx]
    b = [_norm(w["w"]) for w in fresh]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            if a[blk.a + k]:
                pairs[blk.a + k] = blk.b + k
    rep["matched"] = len(pairs)
    if len(pairs) < max(3, 0.45 * len(idx)):
        return orig, rep  # текст слишком разный — не рискуем
    new = [dict(w) for w in orig]
    times: dict[int, tuple[float, float]] = {k: (fresh[j]["s"], fresh[j]["e"]) for k, j in pairs.items()}
    anchors = sorted(times)
    shifts = []
    for k in range(len(idx)):
        o = orig[idx[k]]
        if k in times:
            s, e = times[k]
        else:
            prev = max((x for x in anchors if x < k), default=None)
            nxt = min((x for x in anchors if x > k), default=None)
            if prev is None or nxt is None:
                # край клипа без опоры — сдвигаем так же, как ближайшее совпавшее слово
                ref = prev if prev is not None else nxt
                d = times[ref][0] - orig[idx[ref]]["s"]
                s, e = o["s"] + d, o["e"] + d
            else:
                # равномерно между соседними совпавшими словами
                t0, t1 = times[prev][1], times[nxt][0]
                n = nxt - prev
                s = t0 + (t1 - t0) * (k - prev - 1) / max(1, n - 1) if n > 1 else t0
                e = max(s + 0.08, t0 + (t1 - t0) * (k - prev) / max(1, n - 1)) if n > 1 else t1
        shifts.append(s - o["s"])
        new[idx[k]] = {**o, "s": round(s, 2), "e": round(max(e, s + 0.05), 2)}
    # порядок и отсутствие наложений
    prev_s = -1e9
    for i in idx:
        if new[i]["s"] < prev_s:
            new[i]["s"] = prev_s
        new[i]["e"] = max(new[i]["e"], new[i]["s"] + 0.05)
        prev_s = new[i]["s"]
    rep.update(applied=True, shift=round(float(np.median(shifts)), 3),
               spread=round(float(np.percentile(np.abs(shifts), 90)), 3))
    return new, rep
