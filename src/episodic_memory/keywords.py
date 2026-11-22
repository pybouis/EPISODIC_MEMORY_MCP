"""Volet lexical de la recherche : mots-clés métier des épisodes et glossaire de synonymes.

L'embedding rapproche mal les sigles et abréviations (« PL », « P&L », « PnL », « compte de
résultat »). Chaque épisode porte donc des mots-clés décrivant son SUJET ; si la question
posée contient l'un d'eux (ou l'un de ses synonymes du glossaire), la similarité vectorielle
reçoit un bonus (voir `EpisodicStore.search`).

Le glossaire est un fichier JSON partagé par tous les épisodes, modifiable à la main :

    {"synonyms": [["P&L", "PnL", "compte de résultat", "profit and loss"], ["BFR", "working capital"]]}
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from pathlib import Path

SEP = "|"  # séparateur des mots-clés dans la métadonnée ChromaDB


def normalize(text: str) -> str:
    """Forme de comparaison : minuscules, sans accents ni ponctuation, pluriels en -s retirés.

    Les sigles en « & » sont soudés : « P&L », « P & L » et « PL » donnent tous « pl ».
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"(?<=\b[a-z0-9])\s*&\s*(?=[a-z0-9]\b)", "", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens)


def clean_keywords(keywords: list[str] | None) -> list[str]:
    """Mots-clés nettoyés, sans doublon (à la normalisation près), dans l'ordre donné."""
    out, seen = [], set()
    for keyword in keywords or []:
        keyword = " ".join(str(keyword).replace(SEP, " ").split())
        norm = normalize(keyword)
        if len(norm) >= 2 and norm not in seen:
            seen.add(norm)
            out.append(keyword)
    return out


def split_keywords(stored: str | None) -> list[str]:
    return [k for k in str(stored or "").split(SEP) if k]


class Glossary:
    """Groupes de termes équivalents, relus quand le fichier change (pas de redémarrage)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._stamp: float | None = None
        self._groups: list[list[str]] = []
        self._index: dict[str, set[str]] = {}

    def _load(self) -> None:
        try:
            stamp = self.path.stat().st_mtime
        except OSError:
            stamp = None
        if stamp == self._stamp:
            return
        groups: list[list[str]] = []
        if stamp is not None:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8")).get("synonyms", [])
                groups = [g for g in (clean_keywords(group) for group in raw) if len(g) > 1]
            except (ValueError, AttributeError, TypeError):
                groups = []  # fichier illisible : on cherche sans glossaire plutôt que d'échouer
        index: dict[str, set[str]] = {}
        for group in groups:
            norms = {normalize(term) for term in group}
            for norm in norms:
                index.setdefault(norm, set()).update(norms)
        self._stamp, self._groups, self._index = stamp, groups, index

    def groups(self) -> list[list[str]]:
        with self._lock:
            self._load()
            return [list(g) for g in self._groups]

    def variants(self, term: str) -> set[str]:
        """Formes normalisées du terme et de ses synonymes."""
        norm = normalize(term)
        with self._lock:
            self._load()
            return {norm} | self._index.get(norm, set())

    def add(self, terms: list[str]) -> list[str]:
        """Déclare ces termes équivalents ; fusionne avec les groupes qui en contiennent déjà un."""
        terms = clean_keywords(terms)
        if len(terms) < 2:
            raise ValueError("Il faut au moins deux termes distincts pour déclarer des synonymes.")
        with self._lock:
            self._load()
            norms = {normalize(t) for t in terms}
            merged, others = list(terms), []
            for group in self._groups:
                if norms & {normalize(t) for t in group}:
                    merged += group
                else:
                    others.append(group)
            merged = clean_keywords(merged)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(
                json.dumps({"synonyms": others + [merged]}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self.path)
            self._stamp = None  # relecture au prochain appel
            return merged


_STOPWORDS = set(
    normalize(
        # mots outils français et anglais, tournures de demande : ils ne disent rien du sujet
        "le la les un une des du de au aux ce cet cette ces mon ma mes ton ta tes son sa ses notre nos votre vos "
        "leur leurs et ou mais donc or ni car que qui quoi dont où quel quelle quels quelles lequel laquelle "
        "à dans par pour sur sous avec sans chez vers entre depuis pendant selon comme en y "
        "je tu il elle on nous vous ils elles me te se moi toi lui eux "
        "est sont était être a ont avoir fait faire fais peux peut pouvez veux voudrais souhaite faut "
        "donne donner donnez montre montrer montrez affiche afficher dis dire calcule calculer analyse analyser "
        "explique expliquer liste lister sors sortir combien comment pourquoi quand "
        "ne pas plus moins très tout tous toute toutes aussi si svp merci bonjour stp "
        "the an of to in on at by for from with without and or but is are was be been do does can could would "
        "please give show me my we you they it its this that these those what which how why when"
    ).split()
)


def content_tokens(text: str) -> list[str]:
    """Mots porteurs de sens d'une question : sans mots outils, lettres isolées ni nombres (années…)."""
    return [t for t in normalize(text).split() if len(t) > 1 and not t.isdigit() and t not in _STOPWORDS]


def coverage(question: str, episode_question: str, keywords: list[str], glossary: Glossary | None = None) -> float:
    """Part (0 à 1) des mots porteurs de sens de la question posée que l'on retrouve dans
    l'épisode : dans sa question, ses mots-clés ou leurs synonymes."""
    wanted = content_tokens(question)
    if not wanted:
        return 0.0
    known = set(normalize(episode_question).split())
    for keyword in keywords:
        for variant in glossary.variants(keyword) if glossary else {normalize(keyword)}:
            known.update(variant.split())
    return sum(t in known for t in wanted) / len(wanted)


def haystack(*texts: str | None) -> str:
    """Texte de la requête, normalisé et bordé d'espaces pour chercher des termes entiers."""
    return " " + normalize(" ".join(t for t in texts if t)) + " "


def match_keywords(query: str, keywords: list[str], glossary: Glossary | None = None) -> list[str]:
    """Mots-clés de l'épisode présents dans la requête (`query` vient de `haystack`), synonymes compris."""
    matched, seen = [], set()
    for keyword in keywords:
        if normalize(keyword) in seen:  # synonyme d'un mot-clé déjà retenu : une seule mention
            continue
        variants = glossary.variants(keyword) if glossary else {normalize(keyword)}
        if any(len(v) >= 2 and f" {v} " in query for v in variants):
            matched.append(keyword)
            seen |= variants
    return matched
