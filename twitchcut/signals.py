"""Сигналы «интересности» и поиск кандидатов.

Принцип: дешёвые сигналы (чат, громкость, клипы зрителей) прогоняются по ВСЕМУ стриму
и сужают многочасовой эфир до 10–35 участков. Дорогие этапы (Whisper, LLM) работают
только с ними — это главный способ экономить время и деньги.

Все сигналы нормируются относительно «фона» самого стрима (скользящая медиана),
поэтому одинаково работают и для стримера с 50 зрителями, и с 50 000.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from .lexicon import CATEGORIES, classify_message, is_noise
from .util import ProgressFn, ffmpeg_bin, log, noop_progress

AUDIO_SR = 8000  # для громкости хватает 8 кГц


# --------------------------------------------------------------------- audio
def audio_loudness(audio_path: Path, duration: float, progress: ProgressFn = noop_progress) -> np.ndarray:
    """Громкость в дБ (RMS) по секундам, потоковое декодирование — память не растёт."""
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(audio_path), "-vn", "-ac", "1",
           "-ar", str(AUDIO_SR), "-f", "s16le", "-"]
    kwargs = {"creationflags": 0x08000000} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **kwargs)
    sec_bytes = AUDIO_SR * 2
    chunk_secs = 120
    out: list[float] = []
    buf = b""
    while True:
        data = p.stdout.read(sec_bytes * chunk_secs)
        if not data:
            break
        buf += data
        n = len(buf) // sec_bytes
        if n:
            arr = np.frombuffer(buf[: n * sec_bytes], dtype=np.int16).astype(np.float32).reshape(n, AUDIO_SR)
            rms = np.sqrt(np.mean(arr ** 2, axis=1) + 1e-9)
            out.extend((20 * np.log10(rms / 32768.0 + 1e-9)).tolist())
            buf = buf[n * sec_bytes:]
        if duration:
            progress("signals", min(len(out) / duration, 0.99) * 0.7, f"Анализ звука: {len(out)//60} мин")
    p.wait()
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------- utilities
def smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) == 0:
        return x.astype(np.float32)
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, k, mode="same").astype(np.float32)


def robust_z(x: np.ndarray, win: int, floor: float) -> np.ndarray:
    """(x - скользящая медиана) / скользящий MAD. Устойчиво к самим пикам."""
    n = len(x)
    if n == 0:
        return x
    win = max(30, min(win, n))
    step = max(1, win // 20)
    centers = np.arange(0, n, step)
    med = np.empty(len(centers), dtype=np.float32)
    mad = np.empty(len(centers), dtype=np.float32)
    half = win // 2
    for i, c in enumerate(centers):
        seg = x[max(0, c - half): min(n, c + half)]
        m = np.median(seg)
        med[i] = m
        mad[i] = np.median(np.abs(seg - m))
    med_full = np.interp(np.arange(n), centers, med)
    mad_full = np.interp(np.arange(n), centers, mad)
    scale = np.maximum(np.maximum(1.4826 * mad_full, 0.25 * np.abs(med_full)), floor)
    return ((x - med_full) / scale).astype(np.float32)


# ---------------------------------------------------------------------- chat
def chat_arrays(chat: list, n: int, delay: float) -> dict[str, np.ndarray]:
    """Посекундные ряды чата, сдвинутые на задержку реакции (чтобы совпасть с событием)."""
    rate = np.zeros(n, dtype=np.float32)
    users: list[set] = [set() for _ in range(n)]
    cats = {c: np.zeros(n, dtype=np.float32) for c in CATEGORIES}
    for t, user, text in chat:
        i = int(t - delay)
        if i < 0 or i >= n or is_noise(user, text):
            continue
        rate[i] += 1
        users[i].add(user)
        for c, w in classify_message(text).items():
            cats[c][i] += w
    uniq = np.array([len(u) for u in users], dtype=np.float32)
    return {"rate": rate, "uniq": uniq, **cats}


# ---------------------------------------------------------------- candidates
def find_candidates(duration: float, loud_db: np.ndarray | None, chat: list | None,
                    viewer_clips: list | None, cfg: dict,
                    progress: ProgressFn = noop_progress) -> tuple[list[dict], dict]:
    c = cfg["candidates"]
    w = c["weights"]
    n = int(math.ceil(duration)) + 1
    delay = float(c["chat_delay_sec"])

    total = np.zeros(n, dtype=np.float32)
    weight_sum = 0.0
    parts: dict[str, np.ndarray] = {}

    ch = None
    if chat:
        ch = chat_arrays(chat, n, delay)
        # уникальные авторы важнее: один спамер не должен делать «пик»
        rate_s = smooth(0.5 * ch["rate"] + 0.5 * ch["uniq"], 10)
        reaction = (ch["funny"] * 1.2 + ch["hype"] + ch["shock"] + ch["fail"] * 0.9
                    + ch["cringe"] * 0.8 + ch["wholesome"] * 0.4)
        react_s = smooth(reaction, 10)
        z_rate = robust_z(rate_s, 600, floor=0.15)
        z_react = robust_z(react_s, 600, floor=0.1)
        parts["chat_rate"] = np.clip(z_rate, -1, 10)
        parts["chat_reaction"] = np.clip(z_react, -1, 10)
        total += w["chat_rate"] * parts["chat_rate"] + w["chat_reaction"] * parts["chat_reaction"]
        weight_sum += w["chat_rate"] + w["chat_reaction"]

    if loud_db is not None and len(loud_db):
        a = np.full(n, float(np.median(loud_db)), dtype=np.float32)
        m = min(n, len(loud_db))
        a[:m] = loud_db[:m]
        a = np.maximum(a, -70)  # тишина не должна давать огромные отрицательные z
        z_audio = robust_z(smooth(a, 4), 300, floor=2.0)
        parts["audio"] = np.clip(z_audio, -1, 8)
        total += w["audio"] * parts["audio"]
        weight_sum += w["audio"]

    if weight_sum:
        total /= weight_sum

    if viewer_clips:
        bonus = np.zeros(n, dtype=np.float32)
        tt = np.arange(n, dtype=np.float32)
        for vc in viewer_clips:
            center = vc["offset"] + min(vc.get("duration", 30), 30) * 0.65
            amp = 2.0 + 0.4 * math.log1p(vc.get("views", 0))
            bonus += amp * np.exp(-0.5 * ((tt - center) / 10.0) ** 2)
        parts["viewer_clips"] = bonus
        total += w["viewer_clips"] * bonus

    score = smooth(total, 3)

    # область поиска: пропускаем «приветствия» в начале и прощание в конце
    skip_s = min(c["skip_start_sec"], duration * 0.08)
    skip_e = min(c["skip_end_sec"], duration * 0.05)
    lo, hi = int(skip_s), int(max(skip_s + 1, duration - skip_e))
    hours = duration / 3600
    k = int(np.clip(round(c["per_hour"] * hours), c["min"], c["max"]))
    gap = int(c["min_gap_sec"])

    masked = np.full(n, -np.inf, dtype=np.float32)
    masked[lo:hi] = score[lo:hi]
    picks: list[int] = []
    for _ in range(k):
        i = int(np.argmax(masked))
        if not np.isfinite(masked[i]):
            break
        picks.append(i)
        masked[max(0, i - gap): i + gap + 1] = -np.inf
    picks.sort()

    cands = []
    for idx, t in enumerate(picks, 1):
        start = max(0.0, t - c["context_before"])
        end = min(duration, t + c["context_after"])
        cat_share = {}
        if ch is not None:
            sl = slice(max(0, t - 5), min(n, t + 20))
            sums = {cat: float(ch[cat][sl].sum()) for cat in CATEGORIES}
            tot = sum(sums.values())
            if tot > 0:
                cat_share = {k2: round(v / tot, 2) for k2, v in sorted(sums.items(), key=lambda kv: -kv[1]) if v > 0}
        info = {
            "id": f"c{idx:02d}",
            "peak": float(t),
            "start": float(start),
            "end": float(end),
            "signal": round(float(score[t]), 3),
            "parts": {k2: round(float(v[t]), 2) for k2, v in parts.items()},
            "reaction": cat_share,
        }
        if ch is not None:
            base = float(np.median(smooth(ch["rate"], 10)[lo:hi]) or 0.01)
            info["chat_ratio"] = round(float(smooth(ch["rate"], 10)[t]) / max(base, 0.01), 1)
            info["chat"] = sample_chat(chat, t + delay - 3, t + delay + 20)
        if loud_db is not None and len(loud_db):
            info["loud_delta_db"] = round(float(np.max(loud_db[max(0, t - 6): t + 4]) - np.median(loud_db)), 1)
        if viewer_clips:
            info["viewer_clips"] = sum(1 for vc in viewer_clips if vc["offset"] - 10 <= t <= vc["offset"] + 45)
        cands.append(info)

    # нормированная сила сигнала 0..1 (для итогового ранжирования)
    if cands:
        s = np.array([x["signal"] for x in cands])
        lo_s, hi_s = float(s.min()), float(s.max())
        for x in cands:
            x["signal_norm"] = round((x["signal"] - lo_s) / (hi_s - lo_s), 3) if hi_s > lo_s else 1.0

    # компактная шкала для графика в интерфейсе (шаг 5 с)
    step = 5
    timeline = [round(float(score[i: i + step].max()), 2) for i in range(0, n, step)]
    log.info("Кандидатов: %d (стрим %.1f ч)", len(cands), hours)
    return cands, {"step": step, "score": timeline}


def sample_chat(chat: list, t0: float, t1: float, limit: int = 22) -> list:
    """Выборка показательных сообщений: сначала эмоциональные, без дублей."""
    rows = [m for m in chat if t0 <= m[0] <= t1 and not is_noise(m[1], m[2])]
    seen, scored = set(), []
    for t, user, text in rows:
        key = text.lower().strip()[:40]
        if key in seen:
            continue
        seen.add(key)
        s = sum(classify_message(text).values())
        scored.append((s, t, user, text[:90]))
    scored.sort(key=lambda r: -r[0])
    chosen = sorted(scored[:limit], key=lambda r: r[1])
    # сколько раз встречались повторяющиеся сообщения — тоже сигнал
    return [[round(t, 1), u, txt] for _, t, u, txt in chosen]
