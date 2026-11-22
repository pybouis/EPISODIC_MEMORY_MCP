"""Sécurité : qui appelle (fournisseur d'identité) et ce qu'il a le droit de faire.

Un seul moteur d'autorisation (`Principal`) et deux fournisseurs d'identité au choix
(`EPISODIC_AUTH_PROVIDER`) :

- `local` : annuaire d'utilisateurs stocké dans la base ChromaDB (collection
  `security_users`), un jeton personnel par utilisateur (seul son hash est stocké) ;
- `pa` : le jeton OAuth Planning Analytics de l'appelant est présenté à l'API REST de PA
  (`ActiveUser`), qui renvoie son identité et ses groupes.

Règles :

| Action            | Admin | Admin de groupe | Utilisateur                          |
|-------------------|-------|-----------------|--------------------------------------|
| chercher / lire   | tout  | ses groupes     | ses groupes                          |
| stocker           | tout  | ses groupes     | ses groupes                          |
| remplacer         | tout  | son groupe      | les épisodes qu'il a créés           |
| supprimer         | tout  | son groupe      | ceux qu'il a créés ET validés        |
| exporter          | tout  | son groupe      | non                                  |
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .config import Settings

USERS_COLLECTION = "security_users"
TOKEN_PREFIX = "emk_"


def norm_group(name: str) -> str:
    return name.strip().upper()


def norm_user(name: str) -> str:
    return name.strip().lower()


def _groups(names) -> frozenset[str]:
    return frozenset(g for g in (norm_group(n) for n in names or []) if g)


class AccessDenied(Exception):
    """L'appelant n'a pas le droit de faire cette action."""


class GroupRequired(Exception):
    """L'appelant appartient à plusieurs groupes : il doit préciser lequel."""

    def __init__(self, groups: list[str]):
        super().__init__("Précisez le groupe de l'épisode.")
        self.groups = groups


@dataclass(frozen=True)
class Principal:
    """L'appelant authentifié. `admin_groups` : groupes dont il est l'administrateur."""

    user: str
    is_admin: bool = False
    groups: frozenset[str] = frozenset()
    admin_groups: frozenset[str] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "admin_groups", _groups(self.admin_groups))
        object.__setattr__(self, "groups", _groups(self.groups) | self.admin_groups)

    # ------------------------------------------------------------------ périmètre

    def scope(self) -> list[str] | None:
        """Groupes lisibles ; None = toute la base (administrateur)."""
        return None if self.is_admin else sorted(self.groups)

    def export_scope(self) -> list[str] | None:
        if self.is_admin:
            return None
        if not self.admin_groups:
            raise AccessDenied("L'export est réservé aux administrateurs (de la base ou d'un groupe).")
        return sorted(self.admin_groups)

    def resolve_group(self, requested: str | None, default: str) -> str:
        """Groupe dans lequel stocker un épisode."""
        if requested and requested.strip():
            group = norm_group(requested)
            if not self.is_admin and group not in self.groups:
                raise AccessDenied(f"Vous n'appartenez pas au groupe {group}.")
            return group
        if len(self.groups) == 1:
            return next(iter(self.groups))
        if self.groups:
            raise GroupRequired(sorted(self.groups))
        if self.is_admin:
            return norm_group(default)
        raise AccessDenied("Vous n'appartenez à aucun groupe : stockage impossible.")

    # ------------------------------------------------------------------ épisodes

    def _is(self, name: str) -> bool:
        return bool(name) and norm_user(name) == norm_user(self.user)

    def can_read(self, episode: dict[str, Any]) -> bool:
        return self.is_admin or norm_group(episode.get("group", "")) in self.groups

    def can_replace(self, episode: dict[str, Any]) -> bool:
        if self.is_admin or norm_group(episode.get("group", "")) in self.admin_groups:
            return True
        return self.can_read(episode) and self._is(episode.get("created_by", ""))

    def can_delete(self, episode: dict[str, Any]) -> bool:
        if self.is_admin or norm_group(episode.get("group", "")) in self.admin_groups:
            return True
        # L'auteur ne supprime que ce qu'il a créé ET validé : s'il a été revalidé par
        # quelqu'un d'autre (admin de groupe…), l'épisode ne lui appartient plus seul.
        return (
            self.can_read(episode)
            and self._is(episode.get("created_by", ""))
            and self._is(episode.get("validated_by", ""))
        )

    def describe(self) -> dict[str, Any]:
        role = "admin" if self.is_admin else "group_admin" if self.admin_groups else "user"
        return {
            "user": self.user,
            "role": role,
            "groups": sorted(self.groups),
            "admin_of_groups": sorted(self.admin_groups),
        }


# ============================================================================ annuaire local


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class UserDirectory:
    """Utilisateurs, groupes et rôles, dans la même base ChromaDB que les épisodes."""

    def __init__(self, client):
        # Collection sans recherche vectorielle : un vecteur constant de dimension 1.
        self._col = client.get_or_create_collection(USERS_COLLECTION, embedding_function=None)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lecture

    @staticmethod
    def _record(name: str, meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "user": name,
            "is_admin": bool(meta.get("is_admin")),
            "groups": json.loads(meta.get("groups") or "[]"),
            "admin_of_groups": json.loads(meta.get("admin_groups") or "[]"),
            "active": bool(meta.get("active", True)),
            "has_token": bool(meta.get("token_hash")),
            "created_at": meta.get("created_at", ""),
            "created_by": meta.get("created_by", ""),
        }

    def get(self, user: str) -> dict[str, Any] | None:
        res = self._col.get(ids=[norm_user(user)], include=["documents", "metadatas"])
        return self._record(res["documents"][0], res["metadatas"][0]) if res["ids"] else None

    def list(self) -> list[dict[str, Any]]:
        res = self._col.get(include=["documents", "metadatas"])
        return sorted((self._record(d, m) for d, m in zip(res["documents"], res["metadatas"])), key=lambda r: r["user"])

    def count(self) -> int:
        return self._col.count()

    def principal_for_token(self, token: str) -> Principal | None:
        if not token:
            return None
        res = self._col.get(where={"token_hash": _hash(token)}, include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        rec = self._record(res["documents"][0], res["metadatas"][0])
        if not rec["active"]:
            return None
        return Principal(rec["user"], rec["is_admin"], frozenset(rec["groups"]), frozenset(rec["admin_of_groups"]))

    # ----------------------------------------------------------------- écriture

    def save(
        self,
        user: str,
        *,
        is_admin: bool | None = None,
        groups=None,
        admin_of_groups=None,
        active: bool | None = None,
        created_by: str = "",
    ) -> dict[str, Any]:
        """Crée ou modifie un utilisateur ; `None` conserve la valeur existante."""
        name = user.strip()
        if not name:
            raise ValueError("Le nom d'utilisateur ne peut pas être vide.")
        with self._lock:
            res = self._col.get(ids=[norm_user(name)], include=["documents", "metadatas"])
            meta = dict(res["metadatas"][0]) if res["ids"] else {
                "is_admin": False, "groups": "[]", "admin_groups": "[]", "active": True,
                "token_hash": "", "created_at": _now(), "created_by": created_by,
            }
            if is_admin is not None:
                meta["is_admin"] = bool(is_admin)
            admin_groups = _groups(json.loads(meta["admin_groups"]) if admin_of_groups is None else admin_of_groups)
            member = _groups(json.loads(meta["groups"]) if groups is None else groups) | admin_groups
            meta["groups"] = json.dumps(sorted(member))
            meta["admin_groups"] = json.dumps(sorted(admin_groups))
            if active is not None:
                meta["active"] = bool(active)
            name = res["documents"][0] if res["ids"] else name
            self._col.upsert(ids=[norm_user(name)], documents=[name], embeddings=[[1.0]], metadatas=[meta])
            return self._record(name, meta)

    def new_token(self, user: str) -> str:
        """Génère un nouveau jeton (l'ancien cesse de fonctionner). Il n'est affiché qu'une fois."""
        with self._lock:
            res = self._col.get(ids=[norm_user(user)], include=["metadatas"])
            if not res["ids"]:
                raise KeyError(f"Utilisateur inconnu : {user}")
            token = TOKEN_PREFIX + secrets.token_urlsafe(32)
            meta = dict(res["metadatas"][0]) | {"token_hash": _hash(token)}
            self._col.update(ids=[norm_user(user)], metadatas=[meta])
            return token

    def delete(self, user: str) -> bool:
        with self._lock:
            if not self._col.get(ids=[norm_user(user)], include=[])["ids"]:
                return False
            self._col.delete(ids=[norm_user(user)])
            return True

    # ------------------------------------------- administration déléguée (outils MCP)

    def check_manage(self, actor: Principal, target: dict[str, Any] | None, *, groups, admin_of_groups, is_admin) -> None:
        """Un admin de groupe ne gère que les simples membres de ses propres groupes."""
        if actor.is_admin:
            return
        if not actor.admin_groups:
            raise AccessDenied("La gestion des utilisateurs est réservée aux administrateurs.")
        if is_admin or admin_of_groups:
            raise AccessDenied("Seul un administrateur de la base peut nommer un administrateur.")
        if target and (target["is_admin"] or target["admin_of_groups"]):
            raise AccessDenied("Seul un administrateur de la base peut modifier un administrateur.")
        # Il n'ajoute ou ne retire que ses propres groupes ; les autres appartenances sont intactes.
        before = _groups(target["groups"]) if target else frozenset()
        after = before if groups is None else _groups(groups)
        if (before ^ after) - actor.admin_groups:
            raise AccessDenied(f"Vous ne gérez que les groupes : {', '.join(sorted(actor.admin_groups))}.")

    def check_exclusive(self, actor: Principal, target: dict[str, Any]) -> None:
        """Jeton et désactivation touchent tous les groupes de la cible : il faut les administrer tous."""
        if actor.is_admin:
            return
        if target["is_admin"] or target["admin_of_groups"] or not _groups(target["groups"]) <= actor.admin_groups:
            raise AccessDenied("Cet utilisateur appartient à des groupes que vous n'administrez pas.")


# ============================================================================ fournisseurs


class IdentityProvider(Protocol):
    name: str

    async def authenticate(self, token: str) -> Principal | None: ...


class LocalProvider:
    name = "local"

    def __init__(self, directory: UserDirectory):
        self.directory = directory

    async def authenticate(self, token: str) -> Principal | None:
        import anyio

        return await anyio.to_thread.run_sync(self.directory.principal_for_token, token)


def principal_from_pa(user: str, pa_groups: list[str], settings: Settings) -> Principal:
    """Traduit les groupes Planning Analytics en rôles de la mémoire.

    - un groupe de `pa_admin_groups` -> administrateur de toute la base ;
    - `<préfixe><G><suffixe admin>` -> administrateur du groupe G ;
    - `<préfixe><G>` -> membre du groupe G.
    Avec un préfixe vide, tous les groupes PA (hors groupes système `}…`) sont des groupes
    de la mémoire.
    """
    prefix, suffix = norm_group(settings.pa_group_prefix), norm_group(settings.pa_group_admin_suffix)
    admin_names = {norm_group(g) for g in settings.pa_admin_groups}
    is_admin, member, admin_of = False, set(), set()
    for raw in pa_groups:
        name = norm_group(raw)
        if name in admin_names:
            is_admin = True
            continue
        if not name or name.startswith("}") or not name.startswith(prefix):
            continue
        group = name[len(prefix):]
        if suffix and group.endswith(suffix) and len(group) > len(suffix):
            admin_of.add(group[: -len(suffix)])
        elif group:
            member.add(group)
    return Principal(user, is_admin, frozenset(member), frozenset(admin_of))


class PlanningAnalyticsProvider:
    """Identité et groupes lus dans Planning Analytics avec le jeton de l'appelant."""

    name = "pa"

    def __init__(self, settings: Settings, fetch=None):
        if not settings.pa_url:
            raise ValueError("EPISODIC_PA_URL est obligatoire avec EPISODIC_AUTH_PROVIDER=pa.")
        self.settings = settings
        self._fetch = fetch or self._fetch_active_user
        self._cache: dict[str, tuple[float, Principal | None]] = {}

    async def _fetch_active_user(self, token: str) -> dict[str, Any] | None:
        import httpx

        url = f"{self.settings.pa_url}/ActiveUser?$select=Name,FriendlyName&$expand=Groups($select=Name)"
        async with httpx.AsyncClient(verify=self.settings.pa_verify_tls, timeout=15) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        if resp.status_code in (401, 403):
            return None
        resp.raise_for_status()
        return resp.json()

    async def authenticate(self, token: str) -> Principal | None:
        if not token:
            return None
        key, now = _hash(token), time.monotonic()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            return cached[1]
        data = await self._fetch(token)
        principal = None
        if data and (data.get("FriendlyName") or data.get("Name")):
            groups = [g.get("Name", "") for g in data.get("Groups") or []]
            principal = principal_from_pa(data.get("FriendlyName") or data["Name"], groups, self.settings)
        if len(self._cache) > 1000:
            self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
        # Un refus est gardé moins longtemps : un jeton renouvelé doit repasser vite.
        self._cache[key] = (now + (self.settings.pa_cache_seconds if principal else 30), principal)
        return principal


def build_provider(settings: Settings, client) -> IdentityProvider | None:
    """Fournisseur choisi dans la configuration ; None = pas d'authentification."""
    if settings.auth_provider == "local":
        return LocalProvider(UserDirectory(client))
    if settings.auth_provider == "pa":
        return PlanningAnalyticsProvider(settings)
    return None
