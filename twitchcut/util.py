"""Общие утилиты: логирование, запуск ffmpeg, JSON, форматирование времени."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("twitchcut")
ROOT_DIR = Path(__file__).resolve().parent.parent

# Колбэк прогресса: (стадия, доля 0..1, сообщение)
ProgressFn = Callable[[str, float, str], None]


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # журнал в файл data/logs/twitchcut.log — чтобы потом разобраться, что происходило (например, в эфире)
    try:
        from logging.handlers import RotatingFileHandler
        d = ROOT_DIR / "data" / "logs"
        d.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(d / "twitchcut.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s [%(threadName)s] %(message)s", "%m-%d %H:%M:%S"))
        fh.setLevel(logging.INFO)
        logging.getLogger().addHandler(fh)
    except OSError:
        pass


def noop_progress(stage: str, frac: float, msg: str = "") -> None:
    pass


class TwitchCutError(RuntimeError):
    """Понятная пользователю ошибка (выводится без трейсбека)."""


class Cancelled(TwitchCutError):
    """Пользователь остановил задачу — прерываем долгий шаг (распознавание речи и т.п.)."""


_io_lock = threading.RLock()


def read_json(path: Path, default: Any = None) -> Any:
    for attempt in range(10):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except PermissionError:  # Windows: файл в этот момент заменяется другим потоком
            time.sleep(0.05 * (attempt + 1))
        except json.JSONDecodeError:
            time.sleep(0.05 * (attempt + 1))
    return default


def write_json(path: Path, data: Any) -> None:
    """Атомарная запись. Потокобезопасна; на Windows повторяет попытку, если файл занят читателем."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    with _io_lock:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        last_err = None
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.05 * (attempt + 1))
        try:
            tmp.unlink()
        except OSError:
            pass
        raise last_err  # type: ignore[misc]


def fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


_ffmpeg_cache: dict = {}


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    if name in _ffmpeg_cache:
        return _ffmpeg_cache[name]
    env = os.environ.get("TWITCHCUT_" + name.upper())
    path = env or shutil.which(name)
    if not path and os.name == "nt":
        # winget ставит FFmpeg сюда, но PATH обновляется только в новых окнах
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        cands = [local / "Microsoft" / "WinGet" / "Links" / f"{name}.exe"]
        cands += sorted((local / "Microsoft" / "WinGet" / "Packages").glob(f"Gyan.FFmpeg*/*/bin/{name}.exe"))
        path = next((str(c) for c in cands if c.is_file()), None)
    if not path:
        # Рядом с проектом (например, tools/ffmpeg/bin/ffmpeg.exe)
        root = Path(__file__).resolve().parent.parent
        for cand in root.glob(f"tools/**/{name}*"):
            if cand.stem == name and cand.is_file():
                path = str(cand)
                break
    if not path:
        raise TwitchCutError(
            f"Не найден {name}. Установите FFmpeg (Windows: `winget install Gyan.FFmpeg`) "
            f"и перезапустите терминал."
        )
    _ffmpeg_cache[name] = path
    return path


def run(cmd: list, cwd: Optional[Path] = None, check: bool = True,
        capture: bool = True, input_bytes: Optional[bytes] = None) -> subprocess.CompletedProcess:
    log.debug("RUN %s", " ".join(str(c) for c in cmd))
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    p = subprocess.run(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
        input=input_bytes,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        **kwargs,
    )
    if check and p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", "replace")[-3000:]
        raise TwitchCutError(f"Команда завершилась с ошибкой ({p.returncode}): {cmd[0]}\n{err}")
    return p


def ffprobe_duration(path: Path) -> float:
    p = run([ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    try:
        return float(p.stdout.decode().strip())
    except ValueError:
        return 0.0


def ffprobe_video(path: Path) -> dict:
    p = run([ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", str(path)])
    data = json.loads(p.stdout.decode() or "{}")
    st = (data.get("streams") or [{}])[0]

    def rate(v):
        try:
            a, b = str(v or "0/1").split("/")
            return float(a) / float(b) if float(b) else 0.0
        except ValueError:
            return 0.0
    st["fps"] = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate"))
    return st


def safe_name(s: str, maxlen: int = 60) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    out = "".join("_" if c in bad else c for c in s if ord(c) < 0x2000 or c in "«»—–")  # без эмодзи в имени файла
    out = out.strip(" ._")
    return out[:maxlen] or "clip"
