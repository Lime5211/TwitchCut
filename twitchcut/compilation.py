"""Склейка «Топ-N за стрим» для YouTube Shorts: лучшие клипы задачи одним роликом 9:16 (до 3 минут),
обратный отсчёт — от N-го места к первому, на каждом клипе на 2 секунды крупная плашка «#N».
Отклонённые клипы («Не подходит») не берутся.
"""
from __future__ import annotations

import time
from pathlib import Path

from .render import VCODEC_ARGS, X264_EXTRA, _ass_color, _ass_time, has_audio, pick_encoder
from .util import TwitchCutError, ffmpeg_bin, log, read_json, run, write_json

SHORTS_MAX = 178.0  # YouTube Shorts — до 3 минут


def _duration(p: Path) -> float:
    out = run([ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
              check=False).stdout.decode().strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def pick_clips(job_dir: Path, clips: list[dict], fb: dict, n: int, max_total: float = SHORTS_MAX) -> list[dict]:
    """Лучшие n клипов по оценке, без отклонённых, чтобы вместе влезли в max_total секунд."""
    pool = []
    for c in clips:
        if (fb.get(c.get("file")) or {}).get("status") == "rejected":
            continue
        p = job_dir / c["file"]
        if not p.exists():
            continue
        d = float(c.get("final_duration") or 0) or _duration(p)
        pool.append({**c, "_path": p, "_dur": d})
    pool.sort(key=lambda c: (-(c.get("final_score") or c.get("score") or 0)))
    chosen, total = [], 0.0
    for c in pool:
        if total + c["_dur"] > max_total:
            continue
        chosen.append(c)
        total += c["_dur"]
        if len(chosen) >= n:
            break
    return chosen


def concat_with_numbers(paths: list[Path], out: Path, cfg: dict) -> list[float]:
    """Склеивает ролики по порядку, на первых 2 с каждого — крупная плашка «#N» (обратный отсчёт до #1).
    Возвращает длительности кусков."""
    k = len(paths)
    r = cfg["render"]
    W, H = int(r["width"]), int(r["height"])
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_name(out.stem + "_work")
    work.mkdir(parents=True, exist_ok=True)
    hl = _ass_color(r.get("highlight_color", "#FFE135"))
    font = r.get("font", "Arial Black")
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "", "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
             "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
             "MarginR, MarginV, Encoding",
             f"Style: Num,{font},150,{hl},{hl},&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,10,4,5,0,0,0,1",
             "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    t = 0.0
    durs = []
    for i, p in enumerate(paths):
        d = _duration(p)
        durs.append(d)
        lines.append(f"Dialogue: 0,{_ass_time(t + 0.05)},{_ass_time(t + max(0.6, min(2.0, d - 0.2)))},Num,,0,0,0,,"
                     f"{{\\pos({W // 2},{int(H * 0.09)})\\fad(120,250)\\fscx80\\fscy80\\t(0,150,\\fscx100\\fscy100)}}#{k - i}")
        t += d
    (work / "top.ass").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [ffmpeg_bin(), "-y", "-v", "error"]
    for p in paths:
        cmd += ["-i", str(Path(p).resolve())]
    parts = []
    for i, p in enumerate(paths):
        parts.append(f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
                     f"fps=30,setsar=1,format=yuv420p[v{i}]")
        if has_audio(p):
            parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{durs[i]:.3f}[a{i}]")
        else:
            parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{durs[i]:.3f}[a{i}]")
    parts.append("".join(f"[v{i}][a{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=1[vc][ac]")
    parts.append("[vc]ass=top.ass[vo]")
    enc = pick_encoder(cfg)
    vcodec = VCODEC_ARGS.get(enc) or ["-c:v", "libx264", "-preset", r.get("preset", "veryfast"), "-crf",
                                      str(r.get("crf", 19)), "-profile:v", "high", *X264_EXTRA]
    tmp = out.with_name(out.stem + ".rendering.mp4")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vo]", "-map", "[ac]", *vcodec, "-g", "60", "-r", "30",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp.resolve())]
    try:
        run(cmd, cwd=work)
    except Exception:
        if enc == "libx264":
            raise
        i = cmd.index(vcodec[0])
        cmd[i:i + len(vcodec)] = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", *X264_EXTRA]
        run(cmd, cwd=work)
    tmp.replace(out)
    run([ffmpeg_bin(), "-y", "-v", "error", "-ss", "1", "-i", str(out.resolve()), "-frames:v", "1", "-vf", "scale=360:-2",
         str(out.with_suffix(".jpg").resolve())], check=False)
    for p in work.iterdir():
        p.unlink(missing_ok=True)
    work.rmdir()
    return durs


def build_top(job_dir: Path, clips: list[dict], fb: dict, cfg: dict, n: int = 5) -> dict:
    chosen = pick_clips(job_dir, clips, fb, n)
    if len(chosen) < 2:
        raise TwitchCutError("Для склейки нужно хотя бы 2 готовых клипа (не отклонённых), которые вместе короче 3 минут")
    order = list(reversed(chosen))  # обратный отсчёт: лучший — в конце
    k = len(order)
    out = job_dir / "clips" / f"top_{k}.mp4"
    durs = concat_with_numbers([c["_path"] for c in order], out, cfg)
    thumb = out.with_suffix(".jpg")
    info = {"file": f"clips/{out.name}", "thumb": f"clips/{thumb.name}", "n": k, "duration": round(sum(durs), 1),
            "clips": [{"n": c["n"], "title": c.get("title"), "place": k - i} for i, c in enumerate(order)],
            "created": time.time(), "v": int(time.time())}
    comps = [x for x in (read_json(job_dir / "compilations.json") or []) if x.get("file") != info["file"]]
    write_json(job_dir / "compilations.json", comps + [info])
    log.info("Склейка «Топ-%d»: %s (%.0f с)", k, out.name, info["duration"])
    return info
