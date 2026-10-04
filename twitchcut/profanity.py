"""Поиск мата в транскрипте (по словам с таймкодами) для запикивания и маскировки субтитров.

Русский мат морфологически богат, поэтому используются корни + допустимые приставки,
а не список слов. Есть белый список для ложных срабатываний (страховка, хлеба, употреблять...).
Свои слова можно добавить в data/profanity_extra.txt (по одному корню/регэкспу на строку).
"""
from __future__ import annotations

import re
from pathlib import Path

_PREFIX = r"(?:|по|на|за|вы|у|про|до|об|обо|от|ото|о|под|подъ|из|изъ|раз|разъ|рас|с|съ|въ|в|при|пере|недо|наи|при|не|ни|отъ|долбо|дол[ба]о)"

RU_PATTERNS = [
    r"^" + _PREFIX + r"ху[йеёяию]",              # хуй, нахуя, похуй, охуеть, хуёво
    r"^" + _PREFIX + r"пи[зс]д",                 # пизда, пиздец, распиздяй
    r"^" + _PREFIX + r"[её]б(?:[аеиоуыёя]|л|н|ну|ч|т|ыр|ош|ук|ищ|ись|сь|$)",  # ебать, заебал, выёбывается
    r"^" + _PREFIX + r"[её]бл",
    r"^бл[яЯ]",                                  # бля, блять, блядь
    r"^бл[еэ]ть$",
    r"^сук(?:а|и|у|ой|е|ам|ами|ах|ин|ины|ино)$",
    r"^суч(?:ка|ки|ку|ара|ий|ье|ьи)",
    r"^муд[аои](?:к|ил|зв|х)",                   # мудак, мудила, мудозвон
    r"^пид[оаe]?р",                              # оскорбления
    r"^" + _PREFIX + r"залуп",
    r"^г[ао]нд[оа]н",
    r"^шлюх",
    r"^" + _PREFIX + r"дроч",
]
EN_PATTERNS = [
    r"^(?:mother)?fu+c?k", r"^fck", r"^sh[i1]t", r"^bi+tch", r"^cunt", r"^as+hole", r"^dick(?:head)?$",
    r"^ni+gg", r"^fag", r"^whore", r"^pussy$", r"^bastard",
]
# Ложные срабатывания корней
WHITELIST = re.compile(
    r"^(?:страх|застрах|худ|хул|хутор|хунт|потребл|употребл|угобл|скребл|колебл|ослабл|"
    r"хлеб|себ|веб|небе|ребе|ребё|сукн|сукк|мудр|убл|оскорбл|рубл|тебе|ребят|руб|корабл|сабл|грабл|"
    r"особл|дробл|истребл|блюд|блюз|блок|блог|бланк|блеск|блин|блик|блефе|блефу|блефа|блох|бледн|близ|"
    r"блуд|блюст|блюсти)", re.I)

_LATIN_TO_CYR = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
                               "k": "к", "m": "м", "t": "т", "b": "в", "h": "н"})


def _compile(extra_file: Path | None) -> tuple[list[re.Pattern], list[re.Pattern]]:
    ru = [re.compile(p, re.I) for p in RU_PATTERNS]
    en = [re.compile(p, re.I) for p in EN_PATTERNS]
    if extra_file and extra_file.exists():
        for line in extra_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ru.append(re.compile(line if line.startswith("^") else "^" + re.escape(line), re.I))
    return ru, en


class ProfanityFilter:
    def __init__(self, extra_file: Path | None = None):
        self.ru, self.en = _compile(extra_file)

    @staticmethod
    def normalize(word: str) -> str:
        w = word.lower().replace("ё", "е")
        w = re.sub(r"[^\w*]", "", w)
        return w

    def is_profane(self, word: str) -> bool:
        w = self.normalize(word)
        if not w:
            return False
        if "*" in w and len(w) > 1:
            return True  # Whisper сам зацензурил: «б***ь»
        if re.search(r"[а-я]", w):
            # латинские буквы-двойники в кириллическом слове: «xуй», «бля» с латинской a
            w_cyr = w.translate(_LATIN_TO_CYR)
            if WHITELIST.match(w_cyr):
                return False
            return any(p.search(w_cyr) for p in self.ru)
        return any(p.search(w) for p in self.en)

    @staticmethod
    def mask(word: str) -> str:
        """«блять» → «б***ь», сохраняя пунктуацию вокруг."""
        m = re.match(r"^(\W*)(\w+)(\W*)$", word)
        if not m:
            return "*" * len(word)
        pre, core, post = m.groups()
        if len(core) <= 2:
            masked = core[0] + "*" * (len(core) - 1)
        else:
            masked = core[0] + "*" * (len(core) - 2) + core[-1]
        return pre + masked + post

    def find(self, words: list[dict]) -> list[int]:
        """Индексы матерных слов в списке {w, s, e}."""
        return [i for i, wd in enumerate(words) if self.is_profane(wd["w"])]
