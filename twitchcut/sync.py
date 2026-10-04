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
