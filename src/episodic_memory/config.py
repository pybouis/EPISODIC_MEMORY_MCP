"""Configuration : fichier `.env` puis variables d'environnement (qui priment).

Le `.env` est cherché à côté de l'exécutable (version .exe) ou à la racine du projet ;
`EPISODIC_ENV_FILE` permet d'en désigner un autre.
"""

from __future__ import annotations

import getpass
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Dossier de l'application : celui de l'exe (PyInstaller) ou la racine du projet
# (src/episodic_memory/config.py -> parents[2]).
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_DIR  # ancien nom, conservé

# La base vectorielle vit HORS de OneDrive : la synchronisation de fichiers
# SQLite/HNSW ouverts peut les corrompre. La sauvegarde se fait par export JSON.
DEFAULT_DATA_DIR = Path(r"C:\dev_python\data_episodic_memory_mcp")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TRANSPORTS = ("stdio", "http")
AUTH_PROVIDERS = ("none", "local", "pa")


@dataclass(frozen=True)
class Settings:
    db_dir: Path
    model_cache_dir: Path
    export_dir: Path
    artifacts_dir: Path
    embed_model: str
    threshold: float
    duplicate_threshold: float
    reasoning_max_chars: int
    artifact_max_bytes: int
    user: str
    # --- serveur partagé et sécurité -------------------------------------------------
    transport: str = "stdio"  # EPISODIC_TRANSPORT : stdio (poste local) | http (serveur partagé)
    host: str = "127.0.0.1"  # MCP_HOST
    port: int = 8010  # MCP_PORT
    public_base_url: str = ""  # MCP_PUBLIC_BASE_URL : URL du tunnel (en-tête Host accepté)
    auth_provider: str = "none"  # EPISODIC_AUTH_PROVIDER : none | local | pa
    allow_no_auth: bool = False  # MCP_ALLOW_NO_AUTH : http sans authentification (tests locaux)
    default_group: str = "ADMIN"  # EPISODIC_DEFAULT_GROUP : groupe des épisodes sans groupe précisé
    archive_artifacts: bool = True  # stdio : copie directe des fichiers ; http : envoi par lien signé
    file_link_minutes: int = 30  # EPISODIC_FILE_LINK_MINUTES : validité des liens d'envoi / de téléchargement
    log_dir: Path = DEFAULT_DATA_DIR / "logs"
    # --- fournisseur Planning Analytics ----------------------------------------------
    pa_url: str = ""  # EPISODIC_PA_URL : racine REST de la base TM1 (…/api/<tenant>/v0/tm1/<base>)
    pa_group_prefix: str = "EPISODIC_"  # EPISODIC_PA_GROUP_PREFIX ("" = tous les groupes PA)
    pa_admin_groups: tuple[str, ...] = ("EPISODIC_ADMIN",)  # EPISODIC_PA_ADMIN_GROUPS
    pa_group_admin_suffix: str = "_ADMIN"  # EPISODIC_PA_GROUP_ADMIN_SUFFIX
    pa_cache_seconds: int = 300  # EPISODIC_PA_CACHE_SECONDS
    pa_verify_tls: bool = True  # EPISODIC_PA_VERIFY_TLS
    # --- recherche hybride (vecteur + mots-clés) --------------------------------------
    keyword_bonus: float = 0.25  # EPISODIC_KEYWORD_BONUS : ajouté à la similarité si un mot-clé concorde (0 = désactivé)
    keyword_floor: float = 0.45  # EPISODIC_KEYWORD_FLOOR : similarité vectorielle minimale pour accorder le bonus
    glossary_path: Path = DEFAULT_DATA_DIR / "glossary.json"  # EPISODIC_GLOSSARY : synonymes partagés
    # EPISODIC_KEYWORD_COVERAGE : part minimale des mots de la question retrouvés dans l'épisode pour
    # qu'un mot-clé concordant suffise, même avec une similarité vectorielle faible (> 1 = désactivé)
    keyword_min_coverage: float = 0.75
    # --- tokens -------------------------------------------------------------------------
    # EPISODIC_TRACK_USAGE : mesurer la consommation réelle dans le journal Claude Code (défaut : oui en stdio)
    track_usage: bool = False
    claude_projects_dir: Path = Path.home() / ".claude" / "projects"  # EPISODIC_CLAUDE_PROJECTS_DIR
    chars_per_token: float = 4.0  # EPISODIC_CHARS_PER_TOKEN : estimation de la taille stockée


def _flag(value: str | None, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "oui", "on")


def read_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Valeurs du `.env` complétées (et surchargées) par l'environnement."""
    environ = dict(os.environ) if environ is None else environ
    env_file = Path(environ.get("EPISODIC_ENV_FILE") or APP_DIR / ".env")
    merged: dict[str, str] = {}
    if env_file.is_file():
        from dotenv import dotenv_values

        merged.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    merged.update(environ)
    return merged


def load_settings(environ: dict[str, str] | None = None) -> Settings:
    env = read_env(environ)
    data_dir = Path(env.get("EPISODIC_DATA_DIR") or DEFAULT_DATA_DIR)
    transport = (env.get("EPISODIC_TRANSPORT") or "stdio").strip().lower()
    auth_provider = (env.get("EPISODIC_AUTH_PROVIDER") or "none").strip().lower()
    if transport not in TRANSPORTS:
        raise ValueError(f"EPISODIC_TRANSPORT invalide : {transport!r} (attendu : {', '.join(TRANSPORTS)})")
    if auth_provider not in AUTH_PROVIDERS:
        raise ValueError(f"EPISODIC_AUTH_PROVIDER invalide : {auth_provider!r} (attendu : {', '.join(AUTH_PROVIDERS)})")
    return Settings(
        db_dir=Path(env.get("EPISODIC_DB_DIR") or data_dir / "chroma"),
        model_cache_dir=Path(env.get("EPISODIC_MODEL_CACHE") or data_dir / "models"),
        export_dir=Path(env.get("EPISODIC_EXPORT_DIR") or APP_DIR / "exports"),
        embed_model=env.get("EPISODIC_EMBED_MODEL") or DEFAULT_MODEL,
        # Similarité cosinus minimale pour proposer une question comme « proche ».
        threshold=float(env.get("EPISODIC_THRESHOLD") or "0.75"),
        # Au-delà, une question est considérée comme un doublon lors du stockage.
        duplicate_threshold=float(env.get("EPISODIC_DUPLICATE_THRESHOLD") or "0.95"),
        # Copie d'archive des fichiers produits (Excel, PDF…) rattachés aux épisodes.
        artifacts_dir=Path(env.get("EPISODIC_ARTIFACTS_DIR") or data_dir / "artifacts"),
        # Taille maximale de la synthèse du raisonnement : borne le coût en tokens
        # de sa réutilisation (4000 caractères ≈ 1000 tokens).
        reasoning_max_chars=int(env.get("EPISODIC_REASONING_MAX_CHARS") or "4000"),
        # Au-delà, le fichier n'est pas copié : seul son chemin d'origine est mémorisé.
        artifact_max_bytes=int(float(env.get("EPISODIC_ARTIFACT_MAX_MB") or "50") * 1024 * 1024),
        user=env.get("EPISODIC_USER") or getpass.getuser(),
        transport=transport,
        host=env.get("MCP_HOST") or "127.0.0.1",
        port=int(env.get("MCP_PORT") or env.get("PORT") or 8010),
        public_base_url=(env.get("MCP_PUBLIC_BASE_URL") or "").strip().rstrip("/"),
        auth_provider=auth_provider,
        allow_no_auth=_flag(env.get("MCP_ALLOW_NO_AUTH")),
        default_group=(env.get("EPISODIC_DEFAULT_GROUP") or "ADMIN").strip().upper(),
        # En http, les chemins fournis sont ceux du poste client : le serveur ne peut ni ne doit les
        # lire. Les fichiers lui sont envoyés par un lien signé (voir server.py).
        archive_artifacts=transport == "stdio",
        file_link_minutes=int(env.get("EPISODIC_FILE_LINK_MINUTES") or "30"),
        log_dir=Path(env.get("EPISODIC_LOG_DIR") or data_dir / "logs"),
        pa_url=(env.get("EPISODIC_PA_URL") or "").strip().rstrip("/"),
        pa_group_prefix=env.get("EPISODIC_PA_GROUP_PREFIX", "EPISODIC_").strip(),
        pa_admin_groups=tuple(
            g.strip() for g in (env.get("EPISODIC_PA_ADMIN_GROUPS") or "EPISODIC_ADMIN").split(",") if g.strip()
        ),
        pa_group_admin_suffix=env.get("EPISODIC_PA_GROUP_ADMIN_SUFFIX", "_ADMIN").strip(),
        pa_cache_seconds=int(env.get("EPISODIC_PA_CACHE_SECONDS") or "300"),
        pa_verify_tls=_flag(env.get("EPISODIC_PA_VERIFY_TLS"), default=True),
        # Un mot-clé de l'épisode présent dans la question ajoute ce bonus à la similarité,
        # à condition que la similarité vectorielle atteigne déjà le plancher.
        keyword_bonus=float(env.get("EPISODIC_KEYWORD_BONUS") or "0.25"),
        keyword_floor=float(env.get("EPISODIC_KEYWORD_FLOOR") or "0.45"),
        glossary_path=Path(env.get("EPISODIC_GLOSSARY") or data_dir / "glossary.json"),
        # Question courte formulée autrement (« la trésorerie par Unit » pour un épisode « cash par
        # Unit ») : le vecteur est faible, mais un mot-clé concorde et l'essentiel des mots de la
        # question se retrouve dans l'épisode. La similarité vaut alors cette couverture.
        keyword_min_coverage=float(env.get("EPISODIC_KEYWORD_COVERAGE") or "0.75"),
        # Le journal de conversation est sur le poste de l'utilisateur : lisible seulement en local.
        track_usage=_flag(env.get("EPISODIC_TRACK_USAGE"), default=transport == "stdio"),
        claude_projects_dir=Path(env.get("EPISODIC_CLAUDE_PROJECTS_DIR") or Path.home() / ".claude" / "projects"),
        chars_per_token=float(env.get("EPISODIC_CHARS_PER_TOKEN") or "4"),
    )
