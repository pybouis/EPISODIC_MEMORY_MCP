"""Serveur MCP de mémoire épisodique.

Deux modes (`EPISODIC_TRANSPORT`) :
- `stdio` : poste local, un seul utilisateur, administrateur de sa propre base ;
- `http` : serveur partagé (HTTP streamable), chaque appel est authentifié et limité aux
  groupes de l'appelant (voir `security.py`).
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import secrets
import sys
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, load_settings
from .security import (
    AccessDenied,
    GroupRequired,
    IdentityProvider,
    Principal,
    UserDirectory,
    build_provider,
    norm_group,
)
from .store import DuplicateEpisode, EpisodicStore, ReasoningTooLong
from .usage import measure, normalize_reported

INSTRUCTIONS = """\
Mémoire épisodique : base de référence de paires question/réponse VALIDÉES par l'utilisateur, \
retrouvées par similarité sémantique. Elle sert à capitaliser l'expérience d'une conversation à l'autre.

Protocole à suivre pour toute question ou demande de travail de l'utilisateur — analyse, rapport, \
fichier à produire, explication, diagnostic, développement —, même formulée comme un ordre \
(« fais-moi le rapport… »). Seules exceptions : les salutations, les remerciements, les simples \
confirmations et les réponses à une question que tu viens de poser.

1. COMPARER — Avant de répondre, appelle `find_similar_question` avec la question telle que posée. \
Si elle contient un sigle, une abréviation ou un terme métier qui a d'autres formes courantes, passe \
ces formes dans `keywords` (ex. « PL » → ["P&L", "compte de résultat"]).
2. PROPOSER — Si `matches` n'est pas vide, montre à l'utilisateur la ou les questions proches \
(texte exact, similarité, mots-clés concordants `matched_keywords` s'il y en a, validée par qui et \
quand) et demande-lui s'il confirme qu'il s'agit bien \
d'une question proche de la sienne. N'utilise jamais une réponse stockée sans cet accord.
3. INTÉGRER — S'il confirme, appelle `get_validated_episode`. Tu récupères la réponse de référence, \
la synthèse du raisonnement qui y avait mené (`reasoning`) et les fichiers produits (`artifacts`). \
Traite cette synthèse comme l'historique d'une conversation que tu reprends : repars de la démarche, \
des sources et des décisions déjà établies au lieu de tout redécouvrir, évite les impasses signalées, \
et rouvre un fichier archivé s'il peut servir de modèle. Réponds à sa question ACTUELLE en adaptant \
aux différences entre les deux questions, et signale toute contradiction ou information périmée. \
S'il refuse, ou s'il n'y a aucun match, réponds normalement.
4. CAPITALISER — Une fois la réponse finale donnée, propose à l'utilisateur de valider l'épisode pour \
l'ajouter à la base de référence. Un épisode contient : la question, la réponse (le résultat), la \
synthèse de ton raisonnement (`reasoning`, voir `store_validated_episode`), les mots-clés du sujet \
(`keywords`) et les fichiers produits (Excel, PDF…). Montre-lui la synthèse du raisonnement, les \
mots-clés et la liste des fichiers que tu comptes enregistrer. Seulement s'il accepte explicitement, appelle `store_validated_episode` avec \
`confirmed_by_user=true`. Si la réponse améliore un épisode existant qu'il vient d'utiliser, \
demande-lui s'il préfère remplacer cet épisode (`replace_episode_id`) ou en créer un nouveau.

Ne stocke jamais rien sans validation explicite de l'utilisateur dans la conversation.

Droits : chaque épisode appartient à un groupe ; l'utilisateur ne voit que ceux de ses groupes (`whoami` donne son rôle et ses groupes). Si un outil répond `forbidden`, explique-lui le refus sans chercher à le contourner. Si `store_validated_episode` répond `group_required`, demande-lui dans quel groupe enregistrer l'épisode.
"""

mcp = MCPServer("episodic-memory", instructions=INSTRUCTIONS)

logger = logging.getLogger("episodic_memory")
audit = logging.getLogger("episodic_memory.audit")


class ArtifactRef(BaseModel):
    """Fichier produit pendant le travail (Excel, PDF, script…)."""

    path: str = Field(description="Chemin absolu du fichier, tel qu'il existe au moment du stockage.")
    description: str = Field(default="", description="Ce que contient le fichier, en une phrase.")


class TokenUsage(BaseModel):
    """Tokens consommés par l'IA pour produire la réponse, si le client les connaît."""

    input_tokens: int = Field(default=0, description="Tokens d'entrée non mis en cache.")
    output_tokens: int = Field(default=0, description="Tokens générés (y compris la réflexion).")
    cache_creation_input_tokens: int = Field(default=0, description="Tokens d'entrée écrits en cache.")
    cache_read_input_tokens: int = Field(default=0, description="Tokens d'entrée lus depuis le cache.")
    api_calls: int | None = Field(default=None, description="Nombre d'appels au modèle.")
    model: str | None = Field(default=None, description="Modèle utilisé.")
    subagent_calls: int | None = Field(default=None, description="Dont appels des sous-agents.")
    window: str | None = Field(default=None, description="Fenêtre mesurée (fournie par le hook).")
    started_at: str | None = Field(default=None, description="Début de la fenêtre mesurée.")
    ended_at: str | None = Field(default=None, description="Fin de la fenêtre mesurée.")


_store: EpisodicStore | None = None
_store_lock = threading.Lock()
_directory: UserDirectory | None = None

# Appelant authentifié de la requête HTTP en cours (posé par AuthMiddleware).
_current: ContextVar[Principal | None] = ContextVar("episodic_principal", default=None)


def get_store() -> EpisodicStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = EpisodicStore(load_settings())
        return _store


def current_principal() -> Principal:
    """L'appelant. En stdio (poste local), c'est le propriétaire du processus : administrateur."""
    principal = _current.get()
    if principal is not None:
        return principal
    settings = get_store().settings
    if settings.transport == "stdio" or settings.auth_provider == "none":
        return Principal(settings.user, is_admin=True)
    raise AccessDenied("Appel non authentifié.")


def guarded(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Traduit un refus d'accès en réponse d'outil explicite."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except AccessDenied as exc:
            return {"status": "forbidden", "message": str(exc)}

    return wrapper


def _readable(episode_id: str, principal: Principal) -> dict[str, Any] | None:
    """L'épisode s'il existe ET est lisible par l'appelant (sinon None : on ne révèle pas son existence)."""
    episode = get_store().get(episode_id)
    return episode if episode is not None and principal.can_read(episode) else None


# ------------------------------------------------- fichiers produits (serveur partagé)
# Le serveur ne voit pas les fichiers du poste de l'auteur, et l'assistant ne peut pas les
# faire passer dans un appel d'outil. Ils transitent donc par HTTP, avec un lien à usage
# limité (« ticket ») délivré à un appelant dont les droits viennent d'être vérifiés :
# envoi après `store_validated_episode`, téléchargement après `get_validated_episode`.

_tickets: dict[str, dict[str, Any]] = {}
_tickets_lock = threading.Lock()


def _file_link(kind: str, episode_id: str, name: str, user: str) -> str:
    settings = get_store().settings
    ticket, now = secrets.token_urlsafe(32), time.time()
    with _tickets_lock:
        for key in [k for k, t in _tickets.items() if t["expires"] < now]:
            del _tickets[key]
        _tickets[ticket] = {
            "kind": kind, "episode_id": episode_id, "name": name, "user": user,
            "expires": now + settings.file_link_minutes * 60,
        }
    base = settings.public_base_url or f"http://{settings.host}:{settings.port}"
    return f"{base}/files/{kind}/{ticket}"


def _ticket(kind: str, ticket: str, consume: bool = False) -> dict[str, Any] | None:
    with _tickets_lock:
        found = _tickets.get(ticket)
        if found is None or found["kind"] != kind or found["expires"] < time.time():
            return None
        if consume:
            del _tickets[ticket]
        return found


def _with_downloads(episode: dict[str, Any], principal: Principal) -> dict[str, Any]:
    """Ajoute à chaque fichier reçu par le serveur un lien de téléchargement temporaire."""
    if get_store().settings.archive_artifacts:
        return episode
    for artifact in episode["artifacts"]:
        if artifact.get("stored"):
            artifact["download_url"] = _file_link("dl", episode["id"], artifact["name"], principal.user)
    return episode


def _uploads(episode: dict[str, Any], principal: Principal) -> list[dict[str, str]]:
    """Liens d'envoi des fichiers référencés par l'épisode que le serveur n'a pas encore reçus."""
    if get_store().settings.archive_artifacts:
        return []
    uploads = []
    for artifact in episode["artifacts"]:
        if not artifact.get("stored"):
            url = _file_link("up", episode["id"], artifact["name"], principal.user)
            uploads.append({
                "name": artifact["name"],
                "local_path": artifact["original_path"],
                "upload_url": url,
                "command": f'curl -sS -T "{artifact["original_path"]}" "{url}"',
            })
    return uploads


def _scope(principal: Principal, group: str | None) -> list[str] | None:
    """Périmètre de lecture, éventuellement réduit au groupe demandé."""
    scope = principal.scope()
    if not group or not group.strip():
        return scope
    wanted = norm_group(group)
    return [wanted] if scope is None or wanted in scope else []


@mcp.tool()
@guarded
def find_similar_question(
    question: str,
    top_k: int = 3,
    min_similarity: float | None = None,
    group: str | None = None,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Étape 1 — Compare la question de l'utilisateur aux questions validées de la base de référence.

    Recherche hybride. Renvoie les questions proches (sans leur réponse) avec :
    - `similarity` (0 à 1) : le score comparé au seuil = `vector_similarity` (cosinus des
      embeddings), augmenté d'un bonus quand un mot-clé de l'épisode, ou l'un de ses
      synonymes du glossaire, figure dans la question ; pour une question courte formulée
      autrement, il peut aussi valoir `keyword_coverage`, la part des mots de la question
      retrouvés dans l'épisode (un score élevé avec un `vector_similarity` faible signale
      donc un rapprochement par les mots, à faire confirmer avec d'autant plus de soin) ;
    - `matched_keywords` : les mots-clés de l'épisode qui ont concordé (vide = rapprochement
      purement sémantique). Un rapprochement par mot-clé peut être ambigu (un même sigle
      pour deux notions) : montre ces mots-clés à l'utilisateur ;
    - le groupe, qui a validé et quand, `has_reasoning`, `artifact_names`.
    `matches` vide = aucune question assez proche. La recherche ne porte que sur les groupes
    de l'appelant. Les questions renvoyées doivent être PROPOSÉES à l'utilisateur ; la réponse
    stockée ne s'obtient qu'après son accord, via `get_validated_episode`.

    Args:
        question: la question de l'utilisateur, telle que posée.
        top_k: nombre maximum de questions proches à renvoyer.
        min_similarity: seuil de similarité ; par défaut celui du serveur (0.75).
        group: pour ne chercher que dans un groupe (défaut : tous ceux de l'appelant).
        keywords: autres formes des termes métier de la question (sigle développé,
            abréviation, synonyme, traduction), pour les rapprocher des mots-clés des
            épisodes. Ex. pour « le PL 2022 » : ["P&L", "compte de résultat"]. Uniquement le
            sujet : pas d'entité, d'année ni de devise.
    """
    store = get_store()
    scope = _scope(current_principal(), group)
    matches = store.search(question, top_k=top_k, min_similarity=min_similarity, groups=scope, keywords=keywords)
    return {
        "matches": matches,
        "threshold": store.settings.threshold if min_similarity is None else min_similarity,
        "episodes_in_base": store.count(scope),
    }


@mcp.tool()
@guarded
def get_validated_episode(episode_id: str) -> dict[str, Any]:
    """Étape 3 — Récupère un épisode complet : question, réponse, raisonnement, fichiers.

    À appeler uniquement après que l'utilisateur a confirmé que la question proposée est
    bien proche de la sienne. Contenu :
    - `answer` : la réponse (résultat) validée ;
    - `reasoning` : la synthèse du cheminement qui y avait mené — à traiter comme
      l'historique d'une conversation que l'on reprend (sources, démarche, décisions, pièges) ;
    - `artifacts` : les fichiers produits. En local, `path` est le chemin lisible maintenant
      (copie d'archive), `null` si le fichier n'est plus disponible. Sur un serveur partagé,
      chaque fichier conservé par le serveur porte un `download_url` temporaire : pour
      l'ouvrir, télécharge-le dans un dossier de travail avec
      `curl -sS -o "<dossier>/<name>" "<download_url>"` (ou donne le lien à l'utilisateur) ;
      sans `download_url`, le fichier n'a jamais été envoyé au serveur.
    Incrémente le compteur d'utilisation de l'épisode.
    """
    principal = current_principal()
    if _readable(episode_id, principal) is None:
        return {"status": "not_found", "message": f"Épisode introuvable : {episode_id}"}
    return {"status": "ok", "episode": _with_downloads(get_store().get(episode_id, mark_used=True), principal)}


@mcp.tool()
@guarded
def store_validated_episode(
    question: str,
    answer: str,
    reasoning: str,
    confirmed_by_user: bool,
    artifacts: list[ArtifactRef] | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    replace_episode_id: str | None = None,
    allow_duplicate: bool = False,
    keywords: list[str] | None = None,
    usage: TokenUsage | None = None,
) -> dict[str, Any]:
    """Étape 4 — Enregistre un épisode validé : question, réponse, raisonnement, fichiers.

    N'appeler qu'après accord EXPLICITE de l'utilisateur dans la conversation. La base
    enregistre qui a validé (`validated_by`, l'appelant authentifié) et quand (`validated_at`).

    Args:
        question: la question, reformulée si besoin pour être compréhensible hors contexte.
        answer: la réponse finale validée, complète et autonome (compréhensible sans le
            reste de la conversation). Si le résultat est un fichier (Excel, PDF…), décris
            ici ce qu'il contient et ses conclusions, et passe le fichier dans `artifacts`.
        reasoning: synthèse du cheminement qui a mené à la réponse, à écrire à partir de
            tout l'historique de la conversation (échanges, appels d'outils, essais, corrections).
            C'est une fiche de reprise : ce qu'il faudrait retrouver dans l'historique pour
            refaire ou prolonger ce travail sans tout redécouvrir. Markdown compact, style
            télégraphique, sans recopier la réponse ni raconter la conversation. Rubriques
            (omets celles qui sont vides) :
            - **Contexte** : objectif réel, périmètre, contraintes exprimées par l'utilisateur.
            - **Sources** : fichiers, bases, serveurs, cubes, API, documents consultés (noms et chemins exacts).
            - **Démarche** : étapes clés numérotées, avec les outils, requêtes, formules ou paramètres qui ont marché.
            - **Décisions** : choix et hypothèses retenus, et pourquoi.
            - **Pièges** : erreurs rencontrées, impasses, ce qu'il ne faut pas refaire.
            - **Reprise** : comment adapter à une demande voisine, points restés ouverts.
            Vise 1500 à 2500 caractères ; au-delà de la limite du serveur (4000 par défaut)
            le stockage est refusé avec `reasoning_too_long` : condense et réessaie.
            Pour une question triviale, une ou deux lignes suffisent.
        confirmed_by_user: true seulement si l'utilisateur vient de valider le stockage.
        artifacts: fichiers produits à rattacher à l'épisode. En local, chacun est copié dans
            l'archive de la mémoire. Sur un serveur partagé, la réponse contient `uploads` :
            pour CHAQUE entrée, exécute sa `command` (curl) sur le poste de l'utilisateur
            pour envoyer le fichier au serveur, où tout le groupe pourra le télécharger ;
            vérifie que chaque envoi répond `"status": "stored"` et signale tout échec.
            En remplacement d'un épisode, ne rien passer conserve les fichiers existants.
        group: groupe propriétaire de l'épisode. Inutile si l'appelant n'a qu'un groupe ;
            s'il en a plusieurs, le serveur répond `group_required` avec la liste : demande
            alors à l'utilisateur lequel choisir. Ignoré en remplacement (le groupe est conservé).
        tags: étiquettes facultatives de classement (domaine, projet, entité, année…). Elles
            n'interviennent pas dans la recherche.
        replace_episode_id: id d'un épisode existant à remplacer par cette nouvelle version.
        allow_duplicate: true pour créer un nouvel épisode même si une question quasi
            identique existe déjà (à ne faire que si l'utilisateur le demande).
        keywords: mots-clés du SUJET de la question, avec toutes leurs formes courantes :
            sigle, abréviation, forme développée, synonyme, traduction. Ex. pour un compte de
            résultat : ["P&L", "PnL", "compte de résultat", "profit and loss"]. Ils servent à
            retrouver l'épisode quand la question future emploie un autre mot que celle-ci.
            N'y mets PAS les paramètres de la demande (entité, année, période, devise, nom de
            cube ou de base) : deux questions sur la même entité mais sur des sujets différents
            seraient rapprochées à tort. 3 à 10 mots-clés. En remplacement d'un épisode, ne
            rien passer conserve les mots-clés existants.
        usage: tokens consommés pour produire la réponse. NE PAS le renseigner sous Claude
            Code en local : le serveur les mesure lui-même dans le journal de la conversation.
            À fournir seulement si ton client te les donne réellement ; n'invente jamais de chiffres.

    La réponse contient `episode.size_tokens` (taille estimée de l'épisode) et
    `episode.usage` (consommation mesurée, `null` si elle n'a pas pu l'être) : donne ces
    chiffres à l'utilisateur.
    """
    if not confirmed_by_user:
        return {
            "status": "refused",
            "message": "Stockage refusé : demande d'abord à l'utilisateur de valider la question et la réponse.",
        }
    principal = current_principal()
    store = get_store()
    if replace_episode_id:
        current = _readable(replace_episode_id, principal)
        if current is None:
            return {"status": "error", "message": f"Épisode introuvable : {replace_episode_id}"}
        if not principal.can_replace(current):
            raise AccessDenied(
                "Vous ne pouvez remplacer que les épisodes que vous avez créés "
                "(ou ceux de votre groupe si vous en êtes l'administrateur)."
            )
        target_group = current["group"]
    else:
        try:
            target_group = principal.resolve_group(group, store.settings.default_group)
        except GroupRequired as exc:
            return {
                "status": "group_required",
                "groups": exc.groups,
                "message": "L'utilisateur appartient à plusieurs groupes : demande-lui dans lequel "
                "enregistrer l'épisode, puis rappelle l'outil avec `group`.",
            }
    measured = _measure_usage(store, question, usage)
    try:
        episode = store.add(
            question,
            answer,
            reasoning=reasoning,
            artifacts=None if artifacts is None else [a.model_dump() for a in artifacts],
            validated_by=principal.user,
            tags=tags,
            replace_episode_id=replace_episode_id,
            allow_duplicate=allow_duplicate,
            group=target_group,
            keywords=keywords,
            usage=measured,
        )
    except DuplicateEpisode as dup:
        return {
            "status": "duplicate_found",
            "existing": dup.existing,
            "message": (
                "Une question quasi identique existe déjà. Demande à l'utilisateur s'il veut la remplacer "
                "(replace_episode_id) ou créer malgré tout un nouvel épisode (allow_duplicate=true)."
            ),
        }
    except ReasoningTooLong as exc:
        return {
            "status": "reasoning_too_long",
            "length": exc.length,
            "max_chars": exc.max_chars,
            "message": (
                f"La synthèse du raisonnement fait {exc.length} caractères (maximum {exc.max_chars}). "
                "Condense-la (garde sources, étapes clés, décisions, pièges) puis rappelle l'outil."
            ),
        }
    except (KeyError, ValueError) as exc:
        return {"status": "error", "message": str(exc.args[0] if exc.args else exc)}
    status = "replaced" if replace_episode_id else "stored"
    audit.info("%s %s groupe=%s par=%s", status, episode["id"], episode["group"], principal.user)
    result: dict[str, Any] = {"status": status, "episode": episode}
    uploads = _uploads(episode, principal)
    if uploads:
        result["uploads"] = uploads
        result["message"] = (
            "Épisode enregistré, mais ses fichiers ne sont pas encore sur le serveur : exécute maintenant la "
            f"`command` de chaque entrée de `uploads` (liens valables {store.settings.file_link_minutes} min) "
            "et vérifie que chaque réponse contient \"status\": \"stored\"."
        )
    return result


def _measure_usage(store: EpisodicStore, question: str, reported: TokenUsage | None) -> dict[str, Any] | None:
    """Consommation à enregistrer : déclarée par le client, sinon lue dans le journal Claude Code."""
    if reported is not None:
        return normalize_reported(reported.model_dump())
    if not store.settings.track_usage:
        return None

    def similarity(q: str, candidates: list[str]) -> list[float]:
        vectors = store.embedder.embed([q, *candidates])
        return [sum(a * b for a, b in zip(vectors[0], v)) for v in vectors[1:]]

    try:
        return measure(store.settings.claude_projects_dir, question, similarity)
    except Exception:  # la mesure ne doit jamais empêcher le stockage
        logger.exception("Mesure des tokens impossible")
        return None


@mcp.tool()
@guarded
def set_episode_keywords(episode_id: str, keywords: list[str], confirmed_by_user: bool) -> dict[str, Any]:
    """Remplace les mots-clés d'un épisode existant, sans modifier le reste (réponse, révision, valideur).

    Sert à enrichir un épisode que la recherche n'a pas retrouvé parce que la question
    employait un autre terme (sigle, synonyme). Mêmes règles que le paramètre `keywords` de
    `store_validated_episode` : le sujet et ses variantes, pas les paramètres. Passe la liste
    COMPLÈTE (les mots-clés actuels sont dans `keywords` de `list_episodes`). N'appeler
    qu'après accord explicite de l'utilisateur. Mêmes droits que pour remplacer l'épisode.
    """
    if not confirmed_by_user:
        return {"status": "refused", "message": "Demande d'abord confirmation à l'utilisateur."}
    principal = current_principal()
    current = _readable(episode_id, principal)
    if current is None:
        return {"status": "not_found", "episode_id": episode_id}
    if not principal.can_replace(current):
        raise AccessDenied(
            "Vous ne pouvez modifier que les épisodes que vous avez créés "
            "(ou ceux de votre groupe si vous en êtes l'administrateur)."
        )
    episode = get_store().set_keywords(episode_id, keywords)
    audit.info("keywords %s groupe=%s par=%s", episode_id, episode["group"], principal.user)
    return {"status": "updated", "episode_id": episode_id, "keywords": episode["keywords"]}


@mcp.tool()
@guarded
def set_episode_usage(episode_id: str, usage: TokenUsage) -> dict[str, Any]:
    """Enregistre les tokens consommés pour produire la réponse d'un épisode qui vient d'être stocké.

    À appeler UNIQUEMENT quand le hook de mémoire épisodique te le demande, juste après
    `store_validated_episode`, en recopiant EXACTEMENT les chiffres qu'il fournit (mesurés dans
    le journal de la conversation sur le poste de l'utilisateur). N'invente jamais de chiffres.
    Pas besoin de redemander l'accord de l'utilisateur : il porte sur l'épisode déjà validé.
    Mêmes droits que pour remplacer l'épisode.
    """
    principal = current_principal()
    current = _readable(episode_id, principal)
    if current is None:
        return {"status": "not_found", "episode_id": episode_id}
    if not principal.can_replace(current):
        raise AccessDenied(
            "Vous ne pouvez modifier que les épisodes que vous avez créés "
            "(ou ceux de votre groupe si vous en êtes l'administrateur)."
        )
    data = usage.model_dump()
    measured = normalize_reported(data) | {
        k: data[k] for k in ("subagent_calls", "window", "started_at", "ended_at") if data.get(k) is not None
    }
    measured["source"] = "client_transcript" if data.get("window") else "reported"
    episode = get_store().set_usage(episode_id, measured)
    audit.info("usage %s total=%s par=%s", episode_id, measured["total_tokens"], principal.user)
    return {"status": "updated", "episode_id": episode_id, "usage": episode["usage"], "size_tokens": episode["size_tokens"]}


@mcp.tool()
@guarded
def add_synonyms(terms: list[str], confirmed_by_user: bool) -> dict[str, Any]:
    """Déclare des termes équivalents dans le glossaire partagé de la recherche.

    Un mot-clé d'épisode est alors retrouvé par n'importe lequel de ses synonymes, pour TOUS
    les épisodes (ex. ["BFR", "besoin en fonds de roulement", "working capital"]). Si l'un des
    termes appartient déjà à un groupe de synonymes, les groupes sont fusionnés. Réservé aux
    administrateurs (de la base ou d'un groupe) ; n'appeler qu'après accord explicite.
    """
    if not confirmed_by_user:
        return {"status": "refused", "message": "Demande d'abord confirmation à l'utilisateur."}
    principal = current_principal()
    if not principal.is_admin and not principal.admin_groups:
        raise AccessDenied("Le glossaire est partagé par tous les groupes : sa modification est réservée aux administrateurs.")
    try:
        group = get_store().glossary.add(terms)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    audit.info("synonyms %s par=%s", group, principal.user)
    return {"status": "updated", "synonyms": group}


@mcp.tool()
@guarded
def list_episodes(limit: int = 20, offset: int = 0, group: str | None = None) -> dict[str, Any]:
    """Liste les épisodes visibles par l'appelant (questions, groupe, valideur, date), du plus récent au plus ancien.

    Chaque épisode porte `tokens` : `size` (tokens estimés de question + réponse + raisonnement,
    coût d'une réutilisation) et `consumed` (tokens réellement consommés par l'IA pour produire
    la réponse ; `null` = non mesuré). `token_totals` cumule ces chiffres sur tout le périmètre.
    """
    store = get_store()
    scope = _scope(current_principal(), group)
    return {
        "total": store.count(scope),
        "token_totals": store.token_totals(scope),
        "episodes": store.list(limit=limit, offset=offset, groups=scope),
    }


@mcp.tool()
@guarded
def delete_episode(episode_id: str, confirmed_by_user: bool) -> dict[str, Any]:
    """Supprime définitivement un épisode. N'appeler qu'après accord explicite de l'utilisateur.

    Autorisé à : l'administrateur, l'administrateur du groupe de l'épisode, et l'auteur
    d'un épisode qu'il a à la fois créé et validé.
    """
    if not confirmed_by_user:
        return {"status": "refused", "message": "Suppression refusée : demande d'abord confirmation à l'utilisateur."}
    principal = current_principal()
    episode = _readable(episode_id, principal)
    if episode is None:
        return {"status": "not_found", "episode_id": episode_id}
    if not principal.can_delete(episode):
        raise AccessDenied(
            "Vous ne pouvez supprimer que les épisodes que vous avez créés et validés vous-même "
            f"(celui-ci : créé par {episode['created_by']}, validé par {episode['validated_by']})."
        )
    deleted = get_store().delete(episode_id)
    audit.info("deleted %s groupe=%s par=%s", episode_id, episode["group"], principal.user)
    return {"status": "deleted" if deleted else "not_found", "episode_id": episode_id}


@mcp.tool()
@guarded
def export_episodes() -> dict[str, Any]:
    """Sauvegarde la base dans un fichier JSON horodaté, sur le serveur (dossier exports/).

    Administrateur : toute la base. Administrateur de groupe : les épisodes de ses groupes.
    """
    principal = current_principal()
    scope = principal.export_scope()
    store = get_store()
    path = store.export_json(groups=scope)
    audit.info("exported %s groupes=%s par=%s", path.name, scope or "tous", principal.user)
    return {"status": "exported", "path": str(path), "episodes": store.count(scope)}


@mcp.tool()
@guarded
def whoami() -> dict[str, Any]:
    """Identité de l'appelant dans la mémoire : utilisateur, rôle, groupes, groupes administrés."""
    settings = get_store().settings
    return current_principal().describe() | {"auth_provider": settings.auth_provider}


# ============================================================ gestion des utilisateurs
# Outils enregistrés seulement avec EPISODIC_AUTH_PROVIDER=local (avec `pa`, utilisateurs
# et groupes se gèrent dans Planning Analytics).


def get_directory() -> UserDirectory:
    global _directory
    if _directory is None:
        _directory = UserDirectory(get_store().client)
    return _directory


@guarded
def list_users() -> dict[str, Any]:
    """Liste les utilisateurs de la mémoire (rôle, groupes). Un administrateur de groupe ne voit que ses membres."""
    principal = current_principal()
    users = get_directory().list()
    if not principal.is_admin:
        if not principal.admin_groups:
            raise AccessDenied("La gestion des utilisateurs est réservée aux administrateurs.")
        users = [u for u in users if set(u["groups"]) & principal.admin_groups]
    return {"users": users}


@guarded
def save_user(
    user: str,
    confirmed_by_user: bool,
    groups: list[str] | None = None,
    admin_of_groups: list[str] | None = None,
    is_admin: bool | None = None,
    active: bool | None = None,
) -> dict[str, Any]:
    """Crée ou modifie un utilisateur de la mémoire. N'appeler qu'après accord explicite.

    Un paramètre omis conserve la valeur existante. À la création, un jeton personnel est
    généré et renvoyé UNE SEULE FOIS (`token`) : à transmettre à l'utilisateur, qui le met
    dans l'en-tête `Authorization: Bearer …` de sa connexion MCP.

    Args:
        user: identifiant (adresse e-mail conseillée).
        groups: liste complète des groupes dont il est membre.
        admin_of_groups: groupes dont il est l'administrateur (réservé à l'administrateur de la base).
        is_admin: administrateur de toute la base (réservé à l'administrateur de la base).
        active: false pour désactiver l'accès sans supprimer l'utilisateur.
    """
    if not confirmed_by_user:
        return {"status": "refused", "message": "Demande d'abord confirmation à l'utilisateur."}
    principal, directory = current_principal(), get_directory()
    target = directory.get(user)
    directory.check_manage(principal, target, groups=groups, admin_of_groups=admin_of_groups, is_admin=is_admin)
    if target and active is not None and active != target["active"]:
        directory.check_exclusive(principal, target)
    try:
        record = directory.save(
            user, is_admin=is_admin, groups=groups, admin_of_groups=admin_of_groups, active=active,
            created_by=principal.user,
        )
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    result: dict[str, Any] = {"status": "updated" if target else "created", "user": record}
    if target is None:
        result["token"] = directory.new_token(record["user"])
    audit.info("user %s %s par=%s", result["status"], record["user"], principal.user)
    return result


@guarded
def reset_user_token(user: str, confirmed_by_user: bool) -> dict[str, Any]:
    """Génère un nouveau jeton pour un utilisateur (l'ancien cesse de fonctionner). Renvoyé une seule fois."""
    if not confirmed_by_user:
        return {"status": "refused", "message": "Demande d'abord confirmation à l'utilisateur."}
    principal, directory = current_principal(), get_directory()
    target = directory.get(user)
    if target is None:
        if not principal.is_admin and not principal.admin_groups:
            raise AccessDenied("La gestion des utilisateurs est réservée aux administrateurs.")
        return {"status": "not_found", "user": user}
    if not principal._is(user):  # chacun peut renouveler son propre jeton
        directory.check_manage(principal, target, groups=None, admin_of_groups=None, is_admin=None)
        directory.check_exclusive(principal, target)
    audit.info("user token reset %s par=%s", target["user"], principal.user)
    return {"status": "token_reset", "user": target["user"], "token": directory.new_token(user)}


def register_user_tools() -> None:
    for fn in (list_users, save_user, reset_user_token):
        mcp.tool()(fn)


# ======================================================================= serveur HTTP


class AuthMiddleware:
    """Middleware ASGI : identifie l'appelant par son jeton (Bearer ou X-API-Key), sauf /health."""

    PUBLIC_PATHS = ("/health",)
    PUBLIC_PREFIXES = ("/files/",)  # protégés par leur ticket, délivré à un appelant authentifié

    def __init__(self, app: Any, provider: IdentityProvider | None):
        self.app = app
        self.provider = provider

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if (
            scope["type"] != "http"
            or self.provider is None
            or scope["path"] in self.PUBLIC_PATHS
            or scope["path"].startswith(self.PUBLIC_PREFIXES)
        ):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        token = headers.get("x-api-key", "").strip()
        authorization = headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        principal = None
        try:
            principal = await self.provider.authenticate(token) if token else None
        except Exception as exc:  # fournisseur d'identité injoignable : on refuse, sans détail au client
            logger.error("Authentification impossible (%s) : %s", self.provider.name, exc)
            await self._reply(scope, receive, send, 503, "identity_provider_unavailable")
            return
        if principal is None:
            # Journal sans les en-têtes : le jeton n'est jamais écrit.
            logger.warning("401 %s %s (Host=%s)", scope.get("method", ""), scope["path"], headers.get("host", ""))
            await self._reply(scope, receive, send, 401, "unauthorized")
            return
        reset = _current.set(principal)
        try:
            await self.app(scope, receive, send)
        finally:
            _current.reset(reset)

    @staticmethod
    async def _reply(scope: dict, receive: Callable, send: Callable, status: int, error: str) -> None:
        from starlette.responses import JSONResponse

        headers = {"WWW-Authenticate": 'Bearer realm="episodic-memory"'} if status == 401 else None
        await JSONResponse({"error": error}, status_code=status, headers=headers)(scope, receive, send)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Any) -> Any:
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "server": "episodic-memory", "version": __version__})


@mcp.custom_route("/files/up/{ticket}", methods=["PUT"])
async def upload_file(request: Any) -> Any:
    """Reçoit le contenu brut d'un fichier produit (`curl -T fichier <lien>`)."""
    import anyio
    from starlette.responses import JSONResponse

    ticket = _ticket("up", request.path_params["ticket"])
    if ticket is None:
        return JSONResponse({"status": "error", "error": "lien d'envoi invalide ou expiré"}, status_code=404)
    store = get_store()
    limit = store.settings.artifact_max_bytes
    incoming = store.settings.artifacts_dir / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    tmp, size = incoming / secrets.token_hex(16), 0
    try:
        with tmp.open("wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    return JSONResponse(
                        {"status": "error", "error": f"fichier trop volumineux (maximum {limit // (1024 * 1024)} Mo)"},
                        status_code=413,
                    )
                out.write(chunk)
        if size == 0:
            return JSONResponse({"status": "error", "error": "fichier vide"}, status_code=400)
        try:
            record = await anyio.to_thread.run_sync(store.attach_artifact, ticket["episode_id"], ticket["name"], tmp)
        except KeyError as exc:
            return JSONResponse({"status": "error", "error": str(exc.args[0])}, status_code=404)
    finally:
        tmp.unlink(missing_ok=True)
    _ticket("up", request.path_params["ticket"], consume=True)  # usage unique
    audit.info("file stored %s/%s (%s octets) par=%s", ticket["episode_id"], ticket["name"], size, ticket["user"])
    return JSONResponse({"status": "stored", "name": record["name"], "size_bytes": record["size_bytes"]})


@mcp.custom_route("/files/dl/{ticket}", methods=["GET"])
async def download_file(request: Any) -> Any:
    from starlette.responses import FileResponse, JSONResponse

    ticket = _ticket("dl", request.path_params["ticket"])
    path = get_store().artifact_file(ticket["episode_id"], ticket["name"]) if ticket else None
    if path is None:
        return JSONResponse({"status": "error", "error": "lien de téléchargement invalide ou expiré"}, status_code=404)
    audit.info("file read %s/%s par=%s", ticket["episode_id"], ticket["name"], ticket["user"])
    return FileResponse(path, filename=ticket["name"])


_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def transport_security(settings: Settings) -> Any:
    """Protection DNS rebinding du SDK MCP, compatible avec un tunnel ou un reverse proxy.

    En écoute locale, le SDK n'accepte que `Host: localhost`/`127.0.0.1` et répond 421 aux
    requêtes relayées par un tunnel (Dev Tunnels, Cloudflare) : l'hôte de
    `MCP_PUBLIC_BASE_URL` est donc ajouté à la liste autorisée.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    if settings.host not in _LOCAL_HOSTS:
        return None
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
    if settings.public_base_url:
        public = urlsplit(settings.public_base_url)
        if public.hostname:
            hosts += [public.netloc, f"{public.hostname}:*"]
            origins.append(f"{public.scheme}://{public.netloc}")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )


def create_app(provider: IdentityProvider | None = None) -> Any:
    """Application ASGI complète : serveur MCP (HTTP streamable, sans état) + authentification."""
    store = get_store()
    settings = store.settings
    if provider is None:
        provider = build_provider(settings, store.client)
    if provider is None and not settings.allow_no_auth:
        raise SystemExit(
            "EPISODIC_AUTH_PROVIDER non défini (local ou pa). Définissez-le, ou MCP_ALLOW_NO_AUTH=true "
            "pour un test strictement local : tout appelant serait alors administrateur."
        )
    if settings.auth_provider == "local":
        register_user_tools()
    app = mcp.streamable_http_app(
        stateless_http=True, json_response=True, transport_security=transport_security(settings), host=settings.host
    )
    return AuthMiddleware(app, provider)


def _setup_logging(settings: Settings) -> Path:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_dir / "episodic_memory_mcp.log"
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    ]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s", handlers=handlers, force=True)
    logging.getLogger("mcp").setLevel(logging.WARNING)  # une ligne par requête en mode sans état
    return log_file


def _warm_up() -> None:
    """Charge la base et le modèle d'embedding en arrière-plan pour un premier appel rapide."""
    try:
        store = get_store()
        warm = getattr(store.embedder, "warm_up", None)
        if warm:
            warm()
    except Exception as exc:  # le vrai message ressortira au premier appel d'outil
        print(f"[episodic-memory] préchargement impossible : {exc}", file=sys.stderr)


def run_http() -> None:
    import uvicorn

    settings = get_store().settings
    log_file = _setup_logging(settings)
    app = create_app()
    logger.info(
        "episodic-memory %s - serveur MCP sur http://%s:%s/mcp (authentification : %s) - log : %s",
        __version__, settings.host, settings.port, settings.auth_provider, log_file,
    )
    logger.info("URL publique (MCP_PUBLIC_BASE_URL) : %s", settings.public_base_url or "(non définie)")
    threading.Thread(target=_warm_up, daemon=True).start()
    uvicorn.run(
        app, host=settings.host, port=settings.port, log_level="warning", proxy_headers=True, forwarded_allow_ips="*"
    )


def main() -> None:
    if load_settings().transport == "http":
        run_http()
        return
    threading.Thread(target=_warm_up, daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
