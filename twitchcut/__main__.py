"""Командная строка.

  python -m twitchcut https://www.twitch.tv/videos/123456789
  python -m twitchcut https://www.twitch.tv/videos/123456789 --clips 5 --llm none
  python -m twitchcut D:\\stream.mp4 --chat D:\\chat.json
  python -m twitchcut web              # веб-интерфейс http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import STAGES, Job
from .util import TwitchCutError, log, setup_logging


def _cli_wait(prompt_path: Path, resp_path: Path) -> None:
    print("\n" + "=" * 70)
    print("РУЧНОЙ РЕЖИМ LLM")
    print(f"1. Откройте файл:  {prompt_path}")
    print("2. Скопируйте всё содержимое в чат с Claude (claude.ai).")
    print(f"3. Ответ Claude (JSON) сохраните в файл:  {resp_path}")
    print("=" * 70)
    while not resp_path.exists():
        input("Когда файл с ответом будет сохранён, нажмите Enter... ")


def main(argv=None) -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "web":
        ap = argparse.ArgumentParser(prog="twitchcut web")
        ap.add_argument("--host", default="127.0.0.1")
        ap.add_argument("--port", type=int, default=8765)
        ap.add_argument("--config", type=Path)
        ap.add_argument("-v", "--verbose", action="store_true")
        a = ap.parse_args(argv[1:])
        setup_logging(a.verbose)
        from .web import serve
        serve(a.host, a.port, a.config)
        return 0

    ap = argparse.ArgumentParser(prog="twitchcut", description="Нарезка лучших моментов стрима в клипы 9:16")
    ap.add_argument("source", help="ссылка на VOD Twitch, ID VOD или путь к видеофайлу")
    ap.add_argument("--chat", help="JSON чата (TwitchDownloaderCLI chatdownload) — для локальных файлов")
    ap.add_argument("--config", type=Path, help="путь к config.yaml")
    ap.add_argument("--clips", type=int, help="сколько клипов сделать")
    ap.add_argument("--llm", choices=["api", "claude-cli", "manual", "none"], help="бэкенд анализа")
    ap.add_argument("--model", help="модель Claude для API (например claude-sonnet-5-5)")
    ap.add_argument("--layout", choices=["auto", "split", "cam", "blur", "crop"])
    ap.add_argument("--no-censor", action="store_true", help="не запикивать мат")
    ap.add_argument("--force", choices=STAGES, help="пересчитать начиная с этой стадии")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    setup_logging(a.verbose)

    ov: dict = {}
    if a.clips:
        ov.setdefault("clips", {})["max_clips"] = a.clips
    if a.llm:
        ov.setdefault("llm", {})["backend"] = a.llm
    if a.model:
        ov.setdefault("llm", {})["model"] = a.model
    if a.layout:
        ov.setdefault("render", {})["layout"] = a.layout
    if a.no_censor:
        ov.setdefault("censor", {})["enabled"] = False
    cfg = load_config(a.config, ov)

    last = {"stage": None, "pct": -1}

    def on_update(st: dict) -> None:
        stage, pct = st.get("stage"), int((st.get("stage_progress") or 0) * 100)
        if stage != last["stage"] or pct >= last["pct"] + 10:
            last.update(stage=stage, pct=pct)
            log.info("[%s %3d%%] %s", stage, pct, st.get("message", ""))

    try:
        job = Job(a.source, cfg, chat_file=a.chat, on_update=on_update, wait_manual=_cli_wait)
        state = job.run(force_from=a.force)
    except TwitchCutError as e:
        log.error("%s", e)
        return 1
    print()
    print(f"Готово! Клипы: {job.dir / 'clips'}")
    for c in state.get("clips", []):
        print(f"  {c['n']:>2}. [{c['score']:>3}] {c['stream_time']:>8} {c.get('final_duration', c['duration']):>5.1f}с  {c['title']}")
        if c.get("censored"):
            print(f"      запикано фрагментов: {len(c['censored'])}")
    u = state.get("llm_usage") or {}
    if u.get("cost_usd") is not None:
        print(f"Claude API: {u.get('input_tokens')} вх. + {u.get('output_tokens')} вых. токенов ≈ ${u['cost_usd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
