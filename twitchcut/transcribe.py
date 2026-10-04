"""Распознавание речи (слова с таймкодами).

Два движка:
- whisper.cpp (Vulkan) — на видеокарте AMD / Intel / NVIDIA, в т.ч. встроенной. Ставится setup_gpu.bat
  в tools/whispercpp.
- faster-whisper — на процессоре (или на NVIDIA через CUDA).
Если доступны оба, весь стрим распознаётся ОДНОВРЕМЕННО видеокартой и процессором: куски по 10 минут
раздаются тому, кто свободен.
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .util import ROOT_DIR, ProgressFn, TwitchCutError, ffmpeg_bin, log, noop_progress, read_json, write_json

SR = 16000
WCPP_DIR = ROOT_DIR / "tools" / "whispercpp"
WCPP_STATUS = ROOT_DIR / "data" / "whispercpp_status.json"
# лучшие модели — первыми
WCPP_MODELS = ["ggml-large-v3-turbo-q5_0.bin", "ggml-large-v3-turbo-q8_0.bin", "ggml-large-v3-turbo.bin",
               "ggml-medium-q5_0.bin", "ggml-medium.bin", "ggml-small-q5_1.bin", "ggml-small.bin",
               "ggml-base-q5_1.bin", "ggml-base.bin"]
# типичные «галлюцинации» Whisper на тишине/музыке
HALLUCINATIONS = re.compile(r"(субтитры (сделал|создавал|делал|подогнал)|продолжение следует\.\.\.|"
                            r"(редактор|корректор) субтитров|dimatorzok|amara\.org)", re.I)
_NOHIDE = {"creationflags": 0x08000000} if os.name == "nt" else {}


def wcpp_exe() -> Path | None:
    for name in ("whisper-cli.exe", "whisper-cli"):
        for p in (WCPP_DIR / name, *WCPP_DIR.glob(f"*/{name}")):
            if p.is_file():
                return p
    return None


def wcpp_model(pref: str = "auto") -> Path | None:
    md = WCPP_DIR / "models"
    if pref and pref != "auto":
        p = Path(pref)
        if not p.is_absolute():
            p = md / (pref if pref.endswith(".bin") else f"ggml-{pref}.bin")
        return p if p.is_file() else None
    for name in WCPP_MODELS:
        if (md / name).is_file():
            return md / name
    others = sorted(md.glob("ggml-*.bin")) if md.is_dir() else []
    others = [o for o in others if "silero" not in o.name]
    return others[0] if others else None


def wcpp_vad_model() -> Path | None:
    md = WCPP_DIR / "models"
    found = sorted(md.glob("ggml-silero*.bin")) if md.is_dir() else []
    return found[-1] if found else None


def wcpp_status() -> dict:
    """Для страницы «Статус»: установлен ли движок для видеокарты, какая модель, какая видеокарта."""
    st = read_json(WCPP_STATUS) or {}
    exe, model = wcpp_exe(), wcpp_model()
    return {"installed": bool(exe and model), "exe": str(exe) if exe else None,
            "model": model.name if model else None, "gpu": st.get("gpu"), "integrated": st.get("integrated"),
            "speed": st.get("speed"), "cpu_speed": st.get("cpu_speed"), "error": st.get("error")}


def _status_update(**kw) -> None:
    try:
        st = read_json(WCPP_STATUS) or {}
        st.update(kw)
        write_json(WCPP_STATUS, st)
    except Exception:
        pass


def clean_words(words: list[dict]) -> list[dict]:
    """Убирает типичные галлюцинации Whisper («Субтитры сделал…», «Продолжение следует…») и повторы-петли."""
    if not words:
        return words
    text = " ".join(w["w"] for w in words)
    if HALLUCINATIONS.search(text):
        out, i = [], 0
        while i < len(words):
            hit = False
            for n in range(6, 1, -1):
                chunk = " ".join(w["w"] for w in words[i:i + n])
                m = HALLUCINATIONS.search(chunk)
                if m and m.start() == 0:
                    # выбрасываем всю фразу до конца предложения
                    j = i
                    while j < len(words) and not words[j]["w"].endswith((".", "!", "?")) and j - i < 10:
                        j += 1
                    i = j + 1
                    hit = True
                    break
            if not hit:
                out.append(words[i])
                i += 1
        words = out
    # петли: одно и то же слово 6+ раз подряд
    out = []
    for w in words:
        if len(out) >= 5 and all(x["w"].lower().strip(".,!?") == w["w"].lower().strip(".,!?") for x in out[-5:]):
            continue
        out.append(w)
    return out


def _segments_from_words(words: list[dict]) -> list[dict]:
    segs, cur = [], []
    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > 0.8 or len(cur) >= 25 or cur[-1]["w"].endswith((".", "!", "?"))):
            segs.append({"s": cur[0]["s"], "e": cur[-1]["e"], "text": " ".join(x["w"] for x in cur)})
            cur = []
        cur.append(w)
    if cur:
        segs.append({"s": cur[0]["s"], "e": cur[-1]["e"], "text": " ".join(x["w"] for x in cur)})
    return segs


class WhisperCpp:
    """whisper.cpp (whisper-cli) — распознавание на видеокарте через Vulkan."""

    def __init__(self, cfg: dict, language: str | None):
        self.cfg = cfg
        self.language = language or "auto"
        self.exe = wcpp_exe()
        self.model = wcpp_model(cfg.get("whispercpp_model", "auto"))
        if not self.exe or not self.model:
            raise TwitchCutError("whisper.cpp не установлен — запустите setup_gpu.bat")
        self.vad = wcpp_vad_model()
        self.gpu = None

    def transcribe_range(self, audio_path: Path, s: float, e: float, threads: int = 4) -> tuple[list, list]:
        tmp = Path(tempfile.mkdtemp(prefix="tc_wcpp_"))
        try:
            wav = tmp / "a.wav"
            p = subprocess.run([ffmpeg_bin(), "-v", "error", "-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", str(audio_path),
                                "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **_NOHIDE)
            if p.returncode != 0 or not wav.exists() or wav.stat().st_size < SR:
                return [], []
            cmd = [str(self.exe), "-m", str(self.model), "-f", str(wav), "-l", self.language, "-oj", "-of", str(tmp / "out"),
                   "-ml", "1", "-sow", "-mc", "0", "-t", str(threads)]
            if self.cfg.get("beam_size"):
                cmd += ["-bs", str(int(self.cfg["beam_size"]))]
            if self.vad:
                cmd += ["--vad", "-vm", str(self.vad)]
            t0 = time.time()
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(self.exe.parent), **_NOHIDE)
            err = (p.stderr or b"").decode("utf-8", "replace")
            if self.gpu is None:
                m = re.search(r"ggml_vulkan: \d+ = (.+?)\s+(?:\(|\|)", err)
                self.gpu = m.group(1).strip() if m else ("" if "no GPU found" in err else None)
                if self.gpu is not None:
                    _status_update(gpu=self.gpu or "не найдена (работает на процессоре)",
                                   integrated=bool(re.search(r"uma: 1", err)), model=self.model.name, error=None)
            out = tmp / "out.json"
            if p.returncode != 0 or not out.exists():
                raise TwitchCutError(f"whisper.cpp завершился с ошибкой {p.returncode}: {err.strip()[-400:]}")
            data = json.loads(out.read_text(encoding="utf-8", errors="replace"))
            took = time.time() - t0
            if took > 1:
                _status_update(speed=round((e - s) / took, 1))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        words: list[dict] = []
        for item in data.get("transcription") or []:
            txt = (item.get("text") or "").strip()
            off = item.get("offsets") or {}
            if not txt or txt.startswith("[") and txt.endswith("]"):
                continue
            ws, we = s + off.get("from", 0) / 1000, s + off.get("to", 0) / 1000
            if words and (not re.search(r"\w", txt) or not (item.get("text") or " ")[0].isspace()) and ws - words[-1]["e"] < 0.3:
                # пунктуация или продолжение слова — приклеиваем к предыдущему
                words[-1]["w"] += txt
                words[-1]["e"] = round(max(words[-1]["e"], we), 2)
                continue
            words.append({"w": txt, "s": round(ws, 2), "e": round(max(we, ws + 0.05), 2), "p": 0.9})
        words = clean_words(words)
        return words, _segments_from_words(words)


def _add_nvidia_dll_dirs() -> None:
    """Windows: подхватить cuBLAS/cuDNN из pip-пакетов nvidia-* (если установлены)."""
    if os.name != "nt":
        return
    for base in sys.path:
        nv = Path(base) / "nvidia"
        if nv.is_dir():
            for bin_dir in nv.glob("*/bin"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except (OSError, AttributeError):
                    pass
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def load_audio(audio_path: Path, start: float, dur: float) -> np.ndarray:
    cmd = [ffmpeg_bin(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(audio_path),
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if p.returncode != 0:
        raise TwitchCutError("ffmpeg не смог прочитать аудио: " + p.stderr.decode("utf-8", "replace")[-500:])
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def merge_windows(cands: list[dict]) -> list[tuple[float, float]]:
    spans = sorted((c["start"], c["end"]) for c in cands)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


class Transcriber:
    def __init__(self, cfg: dict):
        self.cfg = cfg["whisper"]
        lang = cfg.get("language") or "auto"
        self.language = None if lang == "auto" else lang
        self.model = None
        self.device = None
        self.gpu: WhisperCpp | None = None
        self.hybrid = False
        backend = self.cfg.get("backend", "auto")
        if backend in ("auto", "whispercpp"):
            try:
                self.gpu = WhisperCpp(self.cfg, self.language)
            except TwitchCutError as e:
                if backend == "whispercpp":
                    log.warning("%s — распознаю на процессоре", e)
        if self.gpu:
            self.hybrid = bool(self.cfg.get("hybrid", True)) and backend == "auto"
            log.info("Распознавание речи: видеокарта (whisper.cpp, %s)%s", self.gpu.model.name,
                     " + процессор одновременно" if self.hybrid else "")

    def _gpu_threads(self) -> int:
        n = int(self.cfg.get("threads") or 0)
        return n or (3 if self.hybrid else max(2, min(8, (os.cpu_count() or 4) - 1)))

    def _choose(self) -> tuple[str, str, str]:
        device = self.cfg["device"]
        if device == "auto":
            device = "cpu"
            try:
                _add_nvidia_dll_dirs()
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
            except Exception:
                pass
        model = self.cfg["model"]
        if model == "auto":
            if device == "cuda":
                model = "large-v3-turbo"
            else:
                # на процессоре small — разумный баланс скорости и качества; medium точнее, но в 2–3 раза медленнее
                model = "small"
        ct = self.cfg["compute_type"]
        if ct == "auto":
            ct = "float16" if device == "cuda" else "int8"
        return device, model, ct

    def _load(self, force_cpu: bool = False) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TwitchCutError("Не установлен faster-whisper: pip install faster-whisper") from e
        device, model, ct = self._choose()
        if force_cpu:
            device, ct = "cpu", "int8"
            if self.cfg["model"] == "auto":
                model = "small"
        log.info("Whisper: модель %s на %s (%s). Первая загрузка модели может занять несколько минут.",
                 model, device.upper(), ct)
        cpu = os.cpu_count() or 2
        threads = int(self.cfg.get("threads") or 0) or (max(1, cpu - 4) if self.hybrid else max(1, cpu - 1))
        self.model = WhisperModel(model, device=device, compute_type=ct, cpu_threads=threads)
        self.device = device

    def _transcribe_array(self, audio: np.ndarray) -> list:
        beam = self.cfg.get("beam_size") or (5 if self.device == "cuda" else 1)
        segments, _info = self.model.transcribe(
            audio, language=self.language, word_timestamps=True, beam_size=beam,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
        )
        return list(segments)

    def transcribe_range(self, audio_path: Path, s: float, e: float, engine: str = "auto") -> tuple[list, list]:
        """Распознаёт отрезок [s, e] стрима. Возвращает (слова, сегменты) в абсолютном времени.
        engine: auto (видеокарта, если есть) | gpu | cpu."""
        if self.gpu and engine in ("auto", "gpu"):
            try:
                return self.gpu.transcribe_range(audio_path, s, e, self._gpu_threads())
            except Exception as ex:
                log.warning("Распознавание на видеокарте не удалось (%s) — дальше на процессоре", ex)
                _status_update(error=str(ex)[:300])
                self.gpu = None
                self.hybrid = False
                if engine == "gpu":
                    raise
        return self._cpu_range(audio_path, s, e)

    def _cpu_range(self, audio_path: Path, s: float, e: float) -> tuple[list, list]:
        if self.model is None:
            self._load()
        audio = load_audio(audio_path, s, e - s)
        if len(audio) < SR // 2:
            return [], []
        try:
            result = self._transcribe_array(audio)
        except Exception as ex:  # чаще всего — нет CUDA-библиотек
            if self.device == "cuda":
                log.warning("Whisper на GPU не запустился (%s). Переключаюсь на CPU.", ex)
                self._load(force_cpu=True)
                result = self._transcribe_array(audio)
            else:
                raise
        words = []
        for seg in result:
            for wd in seg.words or []:
                txt = wd.word.strip()
                if txt:
                    words.append({"w": txt, "s": round(s + wd.start, 2), "e": round(s + wd.end, 2),
                                  "p": round(float(wd.probability), 2)})
        words = clean_words(words)
        return words, _segments_from_words(words)

    def transcribe_windows(self, audio_path: Path, cands: list[dict],
                           progress: ProgressFn = noop_progress) -> dict:
        """Быстрый режим: только окна вокруг кандидатов."""
        windows = merge_windows(cands)
        total = sum(e - s for s, e in windows) or 1
        done = 0.0
        words: list[dict] = []
        segs: list[dict] = []
        for s, e in windows:
            progress("transcribe", done / total, f"Распознаю речь: {done/60:.0f} из {total/60:.0f} мин")
            w, sg = self.transcribe_range(audio_path, s, e)
            words += w
            segs += sg
            done += e - s
        progress("transcribe", 1.0, "Речь распознана")
        return {"words": words, "segments": segs}

    def transcribe_full(self, audio_path: Path, duration: float, parts_dir: Path,
                        progress: ProgressFn = noop_progress, chunk: float = 600.0) -> dict:
        """Глубокий режим: весь стрим кусками по 10 минут. Каждый кусок сохраняется сразу,
        поэтому после перезапуска распознавание продолжается с места остановки.
        Если есть и видеокарта, и процессор — работают одновременно над разными кусками."""
        parts_dir.mkdir(parents=True, exist_ok=True)
        starts = [i * chunk for i in range(int(math.ceil(duration / chunk)))] or [0.0]
        todo = [s for s in starts if read_json(parts_dir / f"part_{int(s):06d}.json") is None]
        total_left = sum(min(duration, s + chunk) - s for s in todo)
        lock = threading.Lock()
        st = {"done": 0.0, "t0": time.time(), "by": {"gpu": 0.0, "cpu": 0.0}, "dead": {}}

        def report():
            done = (duration - total_left) + st["done"]
            eta = ""
            if st["done"] > 0:
                speed = st["done"] / max(1e-6, time.time() - st["t0"])
                eta = f", осталось ~{max(0.0, (total_left - st['done'])) / speed / 60:.0f} мин"
            who = []
            if self.gpu and "gpu" not in st["dead"]:
                who.append("видеокарта")
            if (not self.gpu or self.hybrid) and "cpu" not in st["dead"]:
                who.append("процессор")
            progress("transcribe", done / max(1, duration),
                     f"Распознаю речь всего стрима ({' + '.join(who)}): {done/60:.0f} из {duration/60:.0f} мин{eta}")

        def do(s: float, engine: str) -> None:
            e = min(duration, s + chunk)
            t1 = time.time()
            # +3 с перекрытия, чтобы не резать слово на стыке кусков
            w, sg = self.transcribe_range(audio_path, s, min(duration, e + 3), engine)
            w = [x for x in w if x["s"] < e]
            sg = [x for x in sg if x["s"] < e]
            write_json(parts_dir / f"part_{int(s):06d}.json", {"words": w, "segments": sg, "engine": engine})
            with lock:
                st["done"] += e - s
                st["by"]["gpu" if engine == "gpu" else "cpu"] += e - s
                if engine == "cpu" and self.hybrid:
                    _status_update(cpu_speed=round((e - s) / max(1e-6, time.time() - t1), 1))
                report()

        if todo:
            report()
        if todo and self.gpu and self.hybrid and len(todo) > 1:
            q: queue.Queue = queue.Queue()
            for s in todo:
                q.put(s)

            dead = st["dead"]

            def worker(engine: str) -> None:
                while True:
                    try:
                        s = q.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        do(s, engine)
                    except Exception as ex:
                        q.put(s)  # кусок доделает другой
                        dead[engine] = ex
                        log.warning("%s выбыл(а) из распознавания: %s", "Видеокарта" if engine == "gpu" else "Процессор", ex)
                        return

            # куски раздаются тому, кто освободился первым
            gpu_t = threading.Thread(target=worker, args=("gpu",), daemon=True)
            gpu_t.start()
            worker("cpu")
            gpu_t.join()
            if not q.empty():
                if "cpu" in dead and "gpu" in dead:
                    raise dead["cpu"]
                worker("cpu" if "cpu" not in dead else "gpu")
                if not q.empty():
                    raise dead.get("cpu") or dead.get("gpu") or TwitchCutError("Распознавание не завершено")
            log.info("Распознавание: видеокарта %.0f мин, процессор %.0f мин", st["by"]["gpu"] / 60, st["by"]["cpu"] / 60)
        else:
            for s in todo:
                do(s, "auto")

        words: list[dict] = []
        segs: list[dict] = []
        for s in starts:
            data = read_json(parts_dir / f"part_{int(s):06d}.json") or {"words": [], "segments": []}
            words += data["words"]
            segs += data["segments"]
        progress("transcribe", 1.0, "Речь распознана")
        return {"words": words, "segments": segs}


def slice_words(words: list[dict], start: float, end: float) -> list[dict]:
    return [w for w in words if w["e"] > start and w["s"] < end]


def slice_segments(segs: list[dict], start: float, end: float) -> list[dict]:
    return [s for s in segs if s["e"] > start and s["s"] < end]
