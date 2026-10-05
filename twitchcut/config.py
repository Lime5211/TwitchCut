"""Конфигурация: значения по умолчанию + config.yaml пользователя."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "workspace": "workspace",          # куда складываются задачи и клипы
    "language": "ru",                  # язык стрима для Whisper: ru / en / auto

    "analysis": {
        # deep — распознаётся ВЕСЬ стрим и Claude ищет моменты по смыслу (истории, новости, мнения);
        # fast — только окна вокруг всплесков чата/звука (быстрее, но пропускает «тихие» темы)
        "mode": "deep",
    },

    "scan": {
        "enabled": True,
        "chunk_minutes": 25,           # сколько минут расшифровки Claude читает за один запрос
        "per_hour": 10,                # сколько фрагментов максимум искать на час
        "max_candidates": 40,          # сколько кандидатов (смысл + чат/звук) отдавать на финальную оценку
        "model": "",                   # модель для просмотра (claude-cli): пусто = как llm.cli_model; haiku — экономнее
    },

    "live": {
        # Прямой эфир для стримеров с включённым «Следить за эфиром»
        "poll_seconds": 120,           # как часто проверять, начался ли эфир / подросла ли запись
        "min_chunk_minutes": 60,       # обрабатывать кусками не короче (меньше — быстрее клипы, больше запросов к Claude)
        "max_chunk_minutes": 60,       # и не длиннее — чтобы эфир не занимал ПК надолго и обычные задачи шли между кусками
        "context_minutes": 5,          # сколько минут ДО нового куска Claude перечитывает для контекста
        "backfill_minutes": 0,         # если подключились посреди эфира — сколько последних минут взять (0 — весь эфир с начала)
        "min_score": 60,               # в эфире монтируем только достаточно сильные моменты
        "max_per_cycle": 6,            # не больше клипов за один кусок
        "fill_min_score": 50,          # если сильных нет — лучший момент с оценкой от 50 ...
        "fill_every_minutes": 20,      # ... монтируется, когда клипов не было дольше 20 минут эфира
    },

    "director": {
        # ИИ-режиссёр: Claude смотрит кадры клипа и решает, когда только вебка, когда вебка + экран
        # и что на экране показать крупно (чат, донат, видео, игра). ~20 тыс. токенов на клип.
        "enabled": True,
        "model": "",                   # пусто = как llm.cli_model (sonnet)
        "max_frames": 16,              # сколько кадров клипа он видит
        "min_shot_seconds": 4.0,       # не переключать кадр чаще
        "verify_cam": True,            # один раз за стрим проверить рамку вебки
    },
    "learn": {
        "enabled": True,               # обновлять «вкус канала» по вашим отметкам (раз в 3 новые отметки)
    },

    "cleanup": {
        "keep_days": 2,                # через сколько дней удалять видео старых задач (статистика остаётся)
    },

    "clips": {
        "max_clips": 8,                # сколько готовых клипов рендерить
        "min_duration": 8,
        "max_duration": 300,           # жёсткий предел (5 мин); реальную длину выбирает Claude по смыслу момента
        "target_duration": 30,         # «идеальная» длина для TikTok
        "min_score": 55,               # «сильные» моменты (0..100)
        "fill_min_score": 40,          # если сильных меньше, чем просили, добираем не ниже этой оценки
    },

    "candidates": {
        "per_hour": 7,                 # сколько кандидатов искать на час стрима
        "min": 8,
        "max": 35,
        "min_gap_sec": 50,             # минимальное расстояние между кандидатами
        "chat_delay_sec": 7,           # на сколько чат отстаёт от события (задержка трансляции + реакция)
        "context_before": 90,          # сколько секунд до пика давать на анализ (завязка)
        "context_after": 35,           # и после (развязка)
        "skip_start_sec": 180,         # пропустить начало стрима (приветствия)
        "skip_end_sec": 60,
        "weights": {
            "chat_rate": 1.0,          # всплеск количества сообщений
            "chat_reaction": 1.4,      # эмоции: смех, хайп, шок, фейл
            "audio": 0.8,              # громкость/крик стримера
            "viewer_clips": 1.5,       # зрители сами нарезали клип в этом месте
        },
    },

    "whisper": {
        "model": "auto",               # auto | tiny | base | small | medium | large-v3 | large-v3-turbo
        "device": "auto",              # auto | cuda | cpu
        "compute_type": "auto",
        "beam_size": 0,                # 0 = авто (5 на GPU, 1 на CPU)
        "backend": "auto",             # auto | whispercpp (видеокарта AMD/Intel/NVIDIA через Vulkan) | faster-whisper
        "whispercpp_model": "auto",    # auto = лучшая скачанная модель в tools/whispercpp/models
        "threads": 0,                  # 0 = авто
        "hybrid": True,                # видеокарта и процессор распознают разные куски одновременно
    },

    "llm": {
        # api        — Anthropic API (платно за токены, ключ в ANTHROPIC_API_KEY)
        # claude-cli — локальный Claude Code (`claude -p`), расходует лимиты вашей подписки
        # manual     — TwitchCut сохраняет промпт, вы вставляете его в чат Claude и возвращаете ответ
        # none       — без LLM, только сигналы чата/звука
        "backend": "claude-cli",
        "model": "claude-haiku-4-5-20251001",
        "cli_model": "sonnet",         # модель для claude-cli: haiku (быстро) / sonnet (баланс) / opus (максимум)
        "api_key": "",                 # ключ Anthropic API (или переменная окружения ниже)
        "api_key_env": "ANTHROPIC_API_KEY",
        "batch_size": 12,              # кандидатов в одном запросе
        "max_transcript_chars": 4500,  # обрезка длинных транскриптов на кандидата
        "streamer_context": "",        # кто стример, его мемы, локальные шутки — сильно улучшает выбор
    },

    "censor": {
        "enabled": True,
        # partial — начало слова слышно, остальное глухо (понятно, что сказано, но запикано);
        # muffle — всё слово глухо; mute — тишина; beep — тон
        "mode": "partial",
        "muffle_freq": 320,            # насколько «глухо» (Гц): меньше — глуше
        "muffle_gain": 0.55,           # громкость приглушённой части
        "beep_freq": 420,              # Гц; ~400 = глухой, 1000 = классический ТВ-пик
        "beep_volume": 0.30,
        "pad_ms": 40,                  # запас вокруг слова
        "subtitles": True,             # маскировать мат в субтитрах: Б**ТЬ
        "extra_words_file": "data/profanity_extra.txt",
    },

    "render": {
        "layout": "auto",              # auto | split | blur | crop
        "cam_rect": None,              # [x, y, w, h] вебки в долях кадра (0..1), если авто ошибается
        "width": 1080,
        "height": 1920,
        "fps": "auto",                 # auto: 60, если исходник ~60 к/с (плавнее), иначе 30
        "encoder": "auto",             # auto | libx264 | h264_nvenc
        "crf": 19,
        "preset": "veryfast",
        "quality": 2160,               # макс. высота исходника: берётся лучшее доступное (1440p/4K, если есть)
        "subtitles": True,
        "subtitle_retime": True,       # уточнять время каждого слова субтитров по звуку клипа (процессор, ~15–25 с на клип)
        "subtitle_shift": 0.0,         # ручная поправка субтитров, с: +0.2 — показывать на 0.2 с раньше
        "font": "Arial Black",
        "font_size": 74,
        "highlight_color": "#FFE135",
        "words_per_line": 3,
        "hook_title": False,           # подпись-хук на видео (по умолчанию выкл.: названия только предлагаются)
        "hook_seconds": 3.0,
        "loudnorm": True,
        "trim_pauses": True,           # вырезать «мёртвые» паузы в речи (стример задумался, тишина)
        "pause_min": 0.7,              # для историй/новостей: паузы длиннее этого (с) сокращаются
        "pause_min_comedic": 1.2,      # для смешного: не трогаем короткие паузы перед панчлайном
        "pause_keep": 0.15,            # сколько тишины оставить с каждой стороны, чтобы речь звучала естественно
        "speed": 1.0,                  # ускорение по умолчанию (на странице клипа можно 1.1×–1.5×)
        "zoom": False,                 # «наезд» камеры на громкие реакции (выключен: выглядит странно)
        "zoom_amount": 0.12,           # насколько приближать (0.12 = +12%)
        "cam_height": 820,             # высота вебки сверху, если рамку вебки найти не удалось (из 1920)
        "cam_center": True,            # «вебка + экран»: лицо по центру и крупно, кадр следит за лицом (False — вебка целиком)
        "cam_zoom": 1.0,               # ещё крупнее лицо в зоне вебки: 1.2 = +20%
        "cam_height_min": 560,         # обычно высота зоны вебки подгоняется под её пропорции, чтобы вебка вошла целиком,
        "cam_height_max": 1100,        # но не меньше/не больше этих значений
        "screen_active": 0.05,         # доля экрана, которая должна меняться, чтобы считать, что на экране «что-то идёт»
        "cam_min_seconds": 7.0,        # «только вебка» — не короче этого (экран стоит, стример рассказывает)
        "intro_screen_seconds": 8.0,   # первые секунды клипа показываем экран, чтобы было понятно, о чём речь
        "layout_min_seconds": 5.0,     # раскладка не переключается чаще, чем раз в столько секунд
        "dynamic_layout": True,        # раскладка может меняться внутри клипа (экран → вебка на весь экран)
        "hwdecode": True,              # декодирование исходника видеокартой (-hwaccel auto)
        "fade_out": True,              # мягкое затухание картинки и звука в конце              # выравнивание громкости под TikTok (-14 LUFS)
    },

    "twitch": {
        # Необязательно: для получения клипов, которые сделали зрители (сильный сигнал).
        # https://dev.twitch.tv/console/apps -> Register Application
        "client_id": "",
        "client_secret": "",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None = None, overrides: dict | None = None) -> dict:
    path = path or (ROOT / "config.yaml")
    user = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    ws = Path(cfg["workspace"])
    cfg["workspace"] = str(ws if ws.is_absolute() else ROOT / ws)
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
