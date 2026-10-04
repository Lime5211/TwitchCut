"""Словарь реакций чата: эмоуты и слова → тип эмоции.

Чат — это «живая разметка» зрителей. Важно не только КОЛИЧЕСТВО сообщений,
но и ЧТО они пишут: KEKW = смешно, Pog = круто, monkaS = напряжение, F = провал.
Список можно расширять под конкретного стримера (его 7TV/BTTV-эмоуты).
"""
from __future__ import annotations

import re

CATEGORIES = ("funny", "hype", "shock", "fail", "cringe", "wholesome")

# Эмоуты Twitch/BTTV/FFZ/7TV (сравнение без учёта регистра)
EMOTES: dict[str, str] = {}


def _add(cat: str, *names: str) -> None:
    for n in names:
        EMOTES[n.lower()] = cat


_add("funny", "KEKW", "KEKL", "KEKWait", "LUL", "LULW", "OMEGALUL", "LOLW", "pepeLaugh", "ICANT",
     "Kappa", "4Head", "EleGiggle", "LMAO", "xdd", "xddd", "Xdd", "OMEGADANCE", "KEKG", "peepoGiggles",
     "HAHAHA", "Jebaited", "Clueless", "KappaPride", "LuL", "aRolf", "forsenLUL", "KEKWiggle",
     "Smoge", "dead", "😂", "🤣", "💀")
_add("hype", "Pog", "PogU", "PogChamp", "POGGERS", "Poggers", "PagMan", "PagChomp", "EZ", "EZY",
     "GIGACHAD", "Chad", "LETSGO", "LETSGOOO", "catJAM", "PogBones", "WAYTOODANK", "🔥", "💪",
     "OOOO", "POGCRAZY", "Gigachad", "BASED", "ezClap", "Clap", "monkaChrist")
_add("shock", "monkaS", "monkaW", "monkaGIGA", "D:", "WTF", "WAYTOODANK", "Stare", "PauseChamp",
     "O_o", "o_O", "😳", "😱", "🤯", "Susge", "HUH", "WHAT", "DansGame", "monkaOMEGA", "NOTED", "Aware")
_add("fail", "F", "Sadge", "BibleThump", "PepeHands", "FeelsBadMan", "widepeepoSad", "RIP", "FeelsStrongMan",
     "NotLikeThis", "Deadge", "Despair", "Copium", "COPIUM", "😭", "😢")
_add("cringe", "Cringe", "CRINGE", "WeirdChamp", "FeelsWeirdMan", "monkaHmm", "Weirdge", "ResidentSleeper",
     "🤢", "😬")
_add("wholesome", "<3", "❤", "❤️", "peepoHappy", "FeelsGoodMan", "peepoLove", "widepeepoHappy", "HeyGuys",
     "VirtualHug", "Hug", "catKISS")

# Слова и паттерны (русский + английский чат)
WORD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("funny", re.compile(
        r"(?:[хx][аa]){2,}|(?:[аa][хx]){2,}[аa]?|(?:ha){2,}|(?:ah){2,}|(?:хе){3,}|азаз|lmao|rofl|\blol\b|\bлол\b"
        r"|\bору+\b|\bорн(?:у|ул)|\bкек\b|\)\){2,}|\bxd+\b|\bхд+\b|смешн|угар|ржу|ржака|\bрофл|ржомб|пхпх|пахах",
        re.I)),
    ("hype", re.compile(
        r"\bимба\b|\bнайс\b|\bnice\b|\bклатч|\bclutch|\bкрасава|\bмощ|\bлегенд|\bгоу+\b|\bлетс\s*го|\bеее+\b|\bааа+\b"
        r"|\bpog\b|\bgg\b|\bгг\b|\bwp\b|\bвп\b|\bтащ|\bразъеб|\bвау\b|\bwow\b|\bsheesh|\bкосмос\b|\bжиз+а\b",
        re.I)),
    ("shock", re.compile(
        r"^\?+$|\bчто\?+|\bшо\?+|\bwtf\b|\bвтф\b|\bчего\?+|\bжесть|\bжеск|\bофиге|\bахуе|\bохуе|\bнихуя\s*себе"
        r"|\bwhat\b|\bомг\b|\bomg\b|\bбоже\b|\bкапец|\bпипец|\bжуть|\bстрашн|\bнифига",
        re.I)),
    ("fail", re.compile(
        r"^f$|^ф$|\bрип\b|\brip\b|\bминус\b|\bлох\b|\bпозор|\bфейл|\bfail|\bслил|\bсливает|\bпздц|\bпиздец"
        r"|\bбот\b|\bнуб|\bnoob|\bпотрачено|\bгг\s*вп\b|\bунлак|\bunluck",
        re.I)),
    ("cringe", re.compile(r"\bкринж|\bcringe|\bиспанский\s*стыд|\bстыдно|\bфу+\b", re.I)),
    ("wholesome", re.compile(r"\bмило|\bмилот|\bлюблю|\bобожаю|\bспасибо|\bthank|\blove\b|<3", re.I)),
]

# Сообщения, которые НЕ являются реакцией на момент (иначе начало стрима всегда «пик»)
GREETING_RE = re.compile(
    r"^\s*(?:привет\w*|прив|ку+|здарова|здравствуй\w*|хай|hi|hello|hey|yo|салам\w*|добр\w+\s+\w+|qq|о+,?\s*привет)\b",
    re.I)
BOT_NAMES = {"nightbot", "streamelements", "moobot", "streamlabs", "fossabot", "wizebot", "sery_bot",
             "soundalerts", "botrixoficial", "kofistreambot", "pokemoncommunitygame"}


def classify_message(text: str, emotes: list[str] | None = None) -> dict[str, float]:
    """Возвращает веса эмоций для одного сообщения."""
    scores: dict[str, float] = {}
    tokens = text.split()
    seen_emote = False
    # имена эмоутов уже присутствуют в тексте сообщения как отдельные слова
    for t in tokens:
        cat = EMOTES.get(t.lower())
        if cat:
            # повтор эмоута в одном сообщении усиливает, но с потолком
            scores[cat] = min(scores.get(cat, 0) + 0.6, 2.0)
            seen_emote = True
    for cat, rx in WORD_PATTERNS:
        if rx.search(text):
            scores[cat] = max(scores.get(cat, 0), 1.0)
    # КАПС-сообщения — признак сильной эмоции
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) > 0.8 and not seen_emote:
        scores["hype"] = scores.get("hype", 0) + 0.4
    return scores


def is_noise(user: str, text: str) -> bool:
    if user.lower() in BOT_NAMES:
        return True
    if text.startswith("!"):
        return True  # команды ботов
    if GREETING_RE.match(text):
        return True
    return False


def add_custom(cat: str, names: list[str]) -> None:
    """Эмоуты конкретного канала (из карточки стримера)."""
    for n in names:
        if n:
            EMOTES[n.lower()] = cat
