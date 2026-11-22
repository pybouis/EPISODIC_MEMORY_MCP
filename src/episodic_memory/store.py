"""Base de référence : questions/réponses validées, indexées par embedding (ChromaDB).

Un épisode = la question, la réponse (résultat), la synthèse du raisonnement qui y a
mené, et les fichiers produits (Excel, PDF…), dont une copie d'archive est conservée.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .keywords import SEP, Glossary, clean_keywords, coverage, haystack, match_keywords, split_keywords
from .usage import size_tokens

COLLECTION = "episodes"
# Nombre de voisins vectoriels examinés pour les mots-clés : un épisode dont un mot-clé
# concorde peut ainsi remonter même s'il n'était pas dans les `top_k` premiers. Au-delà de
# ce nombre d'épisodes dans le périmètre, un épisode très éloigné vectoriellement peut échapper
# au rapprochement par mots-clés seuls.
KEYWORD_CANDIDATES = 200


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Embedding local (ONNX, sans clé d'API). Le modèle est chargé au premier usage."""

    def __init__(self, model_name: str, cache_dir: Path):
        self.name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from fastembed import TextEmbedding

                self._cache_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Modèle déjà en cache : aucun accès réseau (démarrage rapide, marche hors ligne).
                    self._model = TextEmbedding(self.name, cache_dir=str(self._cache_dir), local_files_only=True)
                except Exception:
                    # Premier lancement : téléchargement depuis Hugging Face (~220 Mo).
                    self._model = TextEmbedding(self.name, cache_dir=str(self._cache_dir))
            return self._model

    def warm_up(self) -> None:
        self._get()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._get().embed(texts)]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class DuplicateEpisode(Exception):
    def __init__(self, existing: dict[str, Any]):
        super().__init__("Une question quasi identique existe déjà.")
        self.existing = existing


class ReasoningTooLong(Exception):
    def __init__(self, length: int, max_chars: int):
        super().__init__(f"Synthèse du raisonnement trop longue : {length} caractères (maximum {max_chars}).")
        self.length = length
        self.max_chars = max_chars


class EpisodicStore:
    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.embedder = embedder or FastEmbedEmbedder(settings.embed_model, settings.model_cache_dir)
        self.glossary = Glossary(settings.glossary_path)
        settings.db_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.db_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # embedding_function=None : les vecteurs sont toujours fournis par nous.
        existing = {getattr(c, "name", c) for c in self._client.list_collections()}
        if COLLECTION in existing:
            self._col = self._client.get_collection(COLLECTION, embedding_function=None)
        else:
            self._col = self._client.create_collection(
                COLLECTION,
                embedding_function=None,
                metadata={"hnsw:space": "cosine", "embed_model": self.embedder.name},
            )
        stored_model = (self._col.metadata or {}).get("embed_model")
        if stored_model and stored_model != self.embedder.name:
            raise RuntimeError(
                f"La base a été créée avec le modèle d'embedding '{stored_model}', "
                f"mais la configuration demande '{self.embedder.name}'. "
                "Exportez la base, supprimez-la, puis réimportez avec le nouveau modèle."
            )
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lecture

    @property
    def client(self):
        """Client ChromaDB (partagé avec l'annuaire des utilisateurs)."""
        return self._client

    @staticmethod
    def _where(groups: list[str] | None) -> dict[str, Any] | None:
        """Filtre ChromaDB d'un périmètre de groupes ; None = toute la base."""
        if groups is None:
            return None
        return {"group": groups[0]} if len(groups) == 1 else {"group": {"$in": list(groups)}}

    def count(self, groups: list[str] | None = None) -> int:
        """Nombre d'épisodes, limité aux groupes donnés (None = toute la base)."""
        if groups is None:
            return self._col.count()
        if not groups:
            return 0
        return len(self._col.get(where=self._where(groups), include=[])["ids"])

    def search(
        self,
        question: str,
        top_k: int = 3,
        min_similarity: float | None = None,
        groups: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Questions de référence les plus proches, au-dessus du seuil, sans leur réponse.

        Recherche hybride. Quand un mot-clé de l'épisode (ou un synonyme du glossaire) figure
        dans la question ou dans `keywords`, `similarity` est le plus grand de :
        - `vector_similarity` + `keyword_bonus`, si la similarité vectorielle atteint `keyword_floor` ;
        - `keyword_coverage` (part des mots de la question retrouvés dans l'épisode), si elle
          atteint `keyword_min_coverage` — pour les questions courtes formulées autrement.
        Sinon `similarity` = `vector_similarity`. `matched_keywords` liste les mots-clés qui
        ont effectivement relevé le score.

        `groups` restreint la recherche DANS la requête vectorielle : un épisode d'un autre
        groupe n'est jamais comparé, ni compté.
        """
        threshold = self.settings.threshold if min_similarity is None else min_similarity
        n = self.count(groups)
        if n == 0:
            return []
        top_k = max(top_k, 1)
        bonus, floor = self.settings.keyword_bonus, self.settings.keyword_floor
        res = self._col.query(
            query_embeddings=self.embedder.embed([question]),
            n_results=min(max(top_k, KEYWORD_CANDIDATES) if bonus > 0 else top_k, n),
            where=self._where(groups),
            include=["documents", "metadatas", "distances"],
        )
        query_text = haystack(question, *(keywords or []))
        matches = []
        for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            vector = round(1.0 - float(dist), 4)
            similarity, covered = vector, None
            episode_keywords = split_keywords(meta.get("keywords"))
            matched = match_keywords(query_text, episode_keywords, self.glossary) if bonus > 0 else []
            if matched:
                covered = round(coverage(question, doc, episode_keywords, self.glossary), 4)
                if vector >= floor:
                    similarity = round(min(1.0, vector + bonus), 4)
                if covered >= self.settings.keyword_min_coverage:
                    similarity = max(similarity, covered)
                if similarity == vector:  # ni bonus ni couverture : les mots-clés n'ont rien changé
                    matched, covered = [], None
            if similarity >= threshold:
                matches.append(
                    self._summary(id_, doc, meta)
                    | {
                        "similarity": similarity,
                        "vector_similarity": vector,
                        "matched_keywords": matched,
                        "keyword_coverage": covered,
                    }
                )
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches[:top_k]

    def get(self, episode_id: str, mark_used: bool = False) -> dict[str, Any] | None:
        res = self._col.get(ids=[episode_id], include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        meta = dict(res["metadatas"][0])
        if mark_used:
            with self._lock:
                meta["use_count"] = int(meta.get("use_count", 0)) + 1
                meta["last_used_at"] = _now()
                self._col.update(ids=[episode_id], metadatas=[meta])
        return self._full(episode_id, res["documents"][0], meta)

    def list(self, limit: int = 20, offset: int = 0, groups: list[str] | None = None) -> list[dict[str, Any]]:
        if groups is not None and not groups:
            return []
        res = self._col.get(where=self._where(groups), include=["documents", "metadatas"])
        rows = [self._summary(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        rows.sort(key=lambda r: r["validated_at"], reverse=True)
        return rows[offset : offset + limit]

    # ----------------------------------------------------------------- écriture

    def add(
        self,
        question: str,
        answer: str,
        reasoning: str | None = None,
        artifacts: list[dict[str, str]] | None = None,
        validated_by: str | None = None,
        tags: list[str] | None = None,
        replace_episode_id: str | None = None,
        allow_duplicate: bool = False,
        group: str | None = None,
        keywords: list[str] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stocke un épisode validé.

        `usage` : tokens consommés par l'IA pour produire la réponse (voir usage.py). En
        remplacement, `None` conserve la mesure précédente.

        `keywords` : mots-clés du SUJET de la question et leurs variantes (sigles, synonymes),
        utilisés par la recherche hybride. En remplacement, `None` conserve l'existant.

        `group` : groupe propriétaire (défaut : `default_group`). Un remplacement conserve
        le groupe de l'épisode ; la détection de doublon ne regarde que ce groupe.

        `reasoning` : synthèse du cheminement (bornée par reasoning_max_chars).
        `artifacts` : fichiers produits, [{"path": ..., "description": ...}] ; chacun est
        copié dans l'archive de l'épisode. En remplacement, `None` conserve l'existant.

        Lève DuplicateEpisode si une question quasi identique existe (sauf remplacement
        explicite ou allow_duplicate), ReasoningTooLong si la synthèse dépasse la borne.
        """
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            raise ValueError("La question et la réponse ne peuvent pas être vides.")
        if reasoning is not None:
            reasoning = reasoning.strip()
            if len(reasoning) > self.settings.reasoning_max_chars:
                raise ReasoningTooLong(len(reasoning), self.settings.reasoning_max_chars)
        who = (validated_by or self.settings.user).strip()
        group = (group or self.settings.default_group).strip().upper()
        now = _now()
        embedding = self.embedder.embed([question])

        with self._lock:
            if replace_episode_id:
                current = self.get(replace_episode_id)
                if current is None:
                    raise KeyError(f"Épisode introuvable : {replace_episode_id}")
                stored = self._stored_artifacts(replace_episode_id)
                if artifacts is not None:
                    stored = self._archive_artifacts(replace_episode_id, artifacts, previous=stored)
                meta = {
                    "answer": answer,
                    "reasoning": current["reasoning"] if reasoning is None else reasoning,
                    "artifacts": json.dumps(stored, ensure_ascii=False),
                    "validated_by": who,
                    "validated_at": now,
                    "created_by": current["created_by"],
                    "created_at": current["created_at"],
                    "revision": current["revision"] + 1,
                    "use_count": current["use_count"],
                    "last_used_at": current["last_used_at"] or "",
                    "tags": ",".join(tags) if tags is not None else ",".join(current["tags"]),
                    "keywords": SEP.join(current["keywords"] if keywords is None else clean_keywords(keywords)),
                    "group": current["group"],
                    "usage": json.dumps(usage, ensure_ascii=False) if usage else self._raw_usage(replace_episode_id),
                }
                self._col.update(ids=[replace_episode_id], documents=[question], embeddings=embedding, metadatas=[meta])
                return self._full(replace_episode_id, question, meta)

            if not allow_duplicate and self.count([group]):
                res = self._col.query(
                    query_embeddings=embedding,
                    n_results=1,
                    where=self._where([group]),
                    include=["documents", "metadatas", "distances"],
                )
                similarity = round(1.0 - float(res["distances"][0][0]), 4)
                if similarity >= self.settings.duplicate_threshold:
                    raise DuplicateEpisode(
                        self._summary(res["ids"][0][0], res["documents"][0][0], res["metadatas"][0][0])
                        | {"similarity": similarity}
                    )

            episode_id = f"ep_{uuid.uuid4().hex[:12]}"
            stored = self._archive_artifacts(episode_id, artifacts or [])
            meta = {
                "answer": answer,
                "reasoning": reasoning or "",
                "artifacts": json.dumps(stored, ensure_ascii=False),
                "validated_by": who,
                "validated_at": now,
                "created_by": who,
                "created_at": now,
                "revision": 1,
                "use_count": 0,
                "last_used_at": "",
                "tags": ",".join(tags or []),
                "keywords": SEP.join(clean_keywords(keywords)),
                "group": group,
                "usage": json.dumps(usage, ensure_ascii=False) if usage else "",
            }
            try:
                self._col.add(ids=[episode_id], documents=[question], embeddings=embedding, metadatas=[meta])
            except Exception:
                shutil.rmtree(self._episode_dir(episode_id), ignore_errors=True)
                raise
            return self._full(episode_id, question, meta)

    def delete(self, episode_id: str) -> bool:
        with self._lock:
            if self.get(episode_id) is None:
                return False
            self._col.delete(ids=[episode_id])
            shutil.rmtree(self._episode_dir(episode_id), ignore_errors=True)
            return True

    def set_keywords(self, episode_id: str, keywords: list[str]) -> dict[str, Any]:
        """Remplace les mots-clés d'un épisode, sans toucher au reste (ni révision, ni valideur)."""
        with self._lock:
            res = self._col.get(ids=[episode_id], include=["documents", "metadatas"])
            if not res["ids"]:
                raise KeyError(f"Épisode introuvable : {episode_id}")
            meta = dict(res["metadatas"][0]) | {"keywords": SEP.join(clean_keywords(keywords))}
            self._col.update(ids=[episode_id], metadatas=[meta])
            return self._full(episode_id, res["documents"][0], meta)

    def set_usage(self, episode_id: str, usage: dict[str, Any] | None) -> dict[str, Any]:
        """Enregistre (ou efface, avec None) la consommation de tokens d'un épisode."""
        with self._lock:
            res = self._col.get(ids=[episode_id], include=["documents", "metadatas"])
            if not res["ids"]:
                raise KeyError(f"Épisode introuvable : {episode_id}")
            meta = dict(res["metadatas"][0]) | {"usage": json.dumps(usage, ensure_ascii=False) if usage else ""}
            self._col.update(ids=[episode_id], metadatas=[meta])
            return self._full(episode_id, res["documents"][0], meta)

    def token_totals(self, groups: list[str] | None = None) -> dict[str, Any]:
        """Cumul des tokens sur un périmètre : taille stockée et consommation mesurée."""
        out = {
            "episodes": 0,
            "size_tokens": 0,
            "measured_episodes": 0,
            "consumed_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 0,
        }
        if groups is not None and not groups:
            return out
        res = self._col.get(where=self._where(groups), include=["documents", "metadatas"])
        for question, meta in zip(res["documents"], res["metadatas"]):
            out["episodes"] += 1
            out["size_tokens"] += self._size(question, meta)["total"]
            usage = self._usage(meta)
            if usage:
                out["measured_episodes"] += 1
                out["consumed_tokens"] += int(usage.get("total_tokens") or 0)
                out["output_tokens"] += int(usage.get("output_tokens") or 0)
                out["cache_read_tokens"] += int(usage.get("cache_read_input_tokens") or 0)
        return out

    def assign_group(self, group: str, only_missing: bool = True) -> int:
        """Rattache à `group` les épisodes sans groupe (ou tous) ; renvoie le nombre modifié."""
        group = group.strip().upper()
        with self._lock:
            res = self._col.get(include=["metadatas"])
            ids, metas = [], []
            for id_, meta in zip(res["ids"], res["metadatas"]):
                if meta.get("group") == group or (only_missing and meta.get("group")):
                    continue
                ids.append(id_)
                metas.append(dict(meta) | {"group": group})
            if ids:
                self._col.update(ids=ids, metadatas=metas)
            return len(ids)

    # ------------------------------------------------- fichiers produits (archive)

    def _episode_dir(self, episode_id: str) -> Path:
        return self.settings.artifacts_dir / episode_id

    def _stored_artifacts(self, episode_id: str) -> list[dict[str, Any]]:
        res = self._col.get(ids=[episode_id], include=["metadatas"])
        return json.loads(res["metadatas"][0].get("artifacts") or "[]") if res["ids"] else []

    def _archive_artifacts(
        self, episode_id: str, artifacts: list[dict[str, str]], previous: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Copie les fichiers dans l'archive de l'épisode, qui remplace l'archive précédente."""
        if not self.settings.archive_artifacts:
            # Serveur partagé : les chemins sont ceux du poste de l'auteur. Le serveur ne les
            # ouvre pas (il ne les voit pas, et ne doit pas lire ses propres fichiers à la place) :
            # il enregistre la référence, et le fichier lui est ensuite envoyé (attach_artifact).
            shutil.rmtree(self._episode_dir(episode_id), ignore_errors=True)
            records, used = [], set()
            for item in artifacts:
                original = str(item.get("path", "")).strip()
                # Chemin Windows ou POSIX, quel que soit le système du serveur.
                base = re.split(r"[\\/]", original)[-1].strip()
                if not base:
                    continue
                stem, dot, ext = base.rpartition(".")
                name, n = base, 1
                while name.lower() in used:
                    name, n = (f"{stem}_{n}.{ext}" if dot else f"{base}_{n}"), n + 1
                used.add(name.lower())
                records.append(
                    {
                        "name": name,
                        "description": str(item.get("description", "")).strip(),
                        "original_path": original,
                        "stored_path": None,
                        "size_bytes": None,
                    }
                )
            return records
        final = self._episode_dir(episode_id)
        # Un fichier déjà archivé peut être redonné tel quel : on garde son chemin d'origine.
        origin_of = {a["stored_path"]: a["original_path"] for a in previous or [] if a.get("stored_path")}

        sources = []
        for item in artifacts:
            src = Path(str(item.get("path", "")).strip()).expanduser()
            if not src.is_file():
                raise ValueError(f"Fichier produit introuvable : {src}")
            sources.append((src, src.stat().st_size, str(item.get("description", "")).strip()))

        staging = final.with_name(final.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        records, used = [], set()
        if sources:
            staging.mkdir(parents=True)
        for src, size, description in sources:
            record = {
                "name": src.name,
                "description": description,
                "original_path": origin_of.get(str(src), str(src)),
                "stored_path": None,
                "size_bytes": size,
            }
            if size <= self.settings.artifact_max_bytes:
                name, n = src.name, 1
                while name.lower() in used:
                    name, n = f"{src.stem}_{n}{src.suffix}", n + 1
                used.add(name.lower())
                shutil.copy2(src, staging / name)
                record["stored_path"] = str(final / name)
            records.append(record)
        shutil.rmtree(final, ignore_errors=True)
        if sources:
            staging.rename(final)
        return records

    def attach_artifact(self, episode_id: str, name: str, source: Path) -> dict[str, Any]:
        """Range dans l'archive un fichier reçu pour un artefact déjà référencé par l'épisode.

        `source` est un fichier temporaire, déplacé dans l'archive. Lève KeyError si
        l'épisode ou l'artefact n'existe pas.
        """
        with self._lock:
            res = self._col.get(ids=[episode_id], include=["metadatas"])
            if not res["ids"]:
                raise KeyError(f"Épisode introuvable : {episode_id}")
            meta = dict(res["metadatas"][0])
            records = json.loads(meta.get("artifacts") or "[]")
            record = next((a for a in records if a["name"] == name), None)
            if record is None:
                raise KeyError(f"Fichier inconnu pour cet épisode : {name}")
            folder = self._episode_dir(episode_id)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / name
            os.replace(source, target)
            record["stored_path"] = str(target)
            record["size_bytes"] = target.stat().st_size
            meta["artifacts"] = json.dumps(records, ensure_ascii=False)
            self._col.update(ids=[episode_id], metadatas=[meta])
            return record

    def artifact_file(self, episode_id: str, name: str) -> Path | None:
        """Chemin d'archive d'un fichier de l'épisode, s'il a bien été reçu."""
        for a in self._stored_artifacts(episode_id):
            if a["name"] == name and a.get("stored_path") and Path(a["stored_path"]).is_file():
                return Path(a["stored_path"])
        return None

    # ------------------------------------------------------- sauvegarde (JSON)

    def export_json(self, path: Path | None = None, groups: list[str] | None = None) -> Path:
        """Exporte les épisodes (texte et références de fichiers ; pas les fichiers archivés)."""
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "" if groups is None else "_" + "-".join(re.sub(r"\W+", "_", g) for g in groups)
            path = self.settings.export_dir / f"episodes_{stamp}{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if groups is not None and not groups:
            res = {"ids": [], "documents": [], "metadatas": []}
        else:
            res = self._col.get(where=self._where(groups), include=["documents", "metadatas"])
        episodes = [self._full(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        payload = {"exported_at": _now(), "embed_model": self.embedder.name, "episodes": episodes}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def import_json(self, path: Path) -> int:
        """Réimporte un export (les vecteurs sont recalculés). Les ids existants sont écrasés."""
        episodes = json.loads(Path(path).read_text(encoding="utf-8"))["episodes"]
        if not episodes:
            return 0
        keys = ("name", "description", "original_path", "stored_path", "size_bytes")
        with self._lock:
            self._col.upsert(
                ids=[e["id"] for e in episodes],
                documents=[e["question"] for e in episodes],
                embeddings=self.embedder.embed([e["question"] for e in episodes]),
                metadatas=[
                    {
                        "answer": e["answer"],
                        "reasoning": e.get("reasoning", ""),
                        "artifacts": json.dumps(
                            [{k: a.get(k) for k in keys} for a in e.get("artifacts", [])], ensure_ascii=False
                        ),
                        "validated_by": e["validated_by"],
                        "validated_at": e["validated_at"],
                        "created_by": e.get("created_by", e["validated_by"]),
                        "created_at": e.get("created_at", e["validated_at"]),
                        "revision": int(e.get("revision", 1)),
                        "use_count": int(e.get("use_count", 0)),
                        "last_used_at": e.get("last_used_at") or "",
                        "tags": ",".join(e.get("tags", [])),
                        "keywords": SEP.join(clean_keywords(e.get("keywords", []))),
                        "group": (e.get("group") or self.settings.default_group).strip().upper(),
                        "usage": json.dumps(e["usage"], ensure_ascii=False) if e.get("usage") else "",
                    }
                    for e in episodes
                ],
            )
        return len(episodes)

    # ------------------------------------------------------------------ interne

    def _raw_usage(self, episode_id: str) -> str:
        res = self._col.get(ids=[episode_id], include=["metadatas"])
        return str(res["metadatas"][0].get("usage") or "") if res["ids"] else ""

    @staticmethod
    def _usage(meta: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return json.loads(meta.get("usage") or "null")
        except ValueError:
            return None

    def _size(self, question: str, meta: dict[str, Any]) -> dict[str, int]:
        # Calculée à la lecture : toujours juste, y compris pour les épisodes antérieurs.
        return size_tokens(question, meta.get("answer", ""), meta.get("reasoning"), self.settings.chars_per_token)

    def _summary(self, id_: str, question: str, meta: dict[str, Any]) -> dict[str, Any]:
        usage = self._usage(meta)
        return {
            "id": id_,
            "question": question,
            "group": meta.get("group", ""),
            "validated_by": meta.get("validated_by", ""),
            "validated_at": meta.get("validated_at", ""),
            "use_count": int(meta.get("use_count", 0)),
            "tags": [t for t in str(meta.get("tags", "")).split(",") if t],
            "keywords": split_keywords(meta.get("keywords")),
            "has_reasoning": bool(meta.get("reasoning")),
            "artifact_names": [a["name"] for a in json.loads(meta.get("artifacts") or "[]")],
            # size : tokens estimés de question + réponse + raisonnement (coût d'une réutilisation) ;
            # consumed : tokens réellement consommés pour produire la réponse, relectures du cache
            # comprises ; output : dont tokens générés (None = non mesuré).
            "tokens": {
                "size": self._size(question, meta)["total"],
                "consumed": int(usage["total_tokens"]) if usage and usage.get("total_tokens") is not None else None,
                "output": int(usage["output_tokens"]) if usage and usage.get("output_tokens") is not None else None,
            },
        }

    def _full(self, id_: str, question: str, meta: dict[str, Any]) -> dict[str, Any]:
        artifacts = []
        for a in json.loads(meta.get("artifacts") or "[]"):
            if self.settings.archive_artifacts:
                # `path` : chemin utilisable maintenant (copie d'archive, sinon l'original s'il existe encore).
                usable = next(
                    (p for p in (a.get("stored_path"), a.get("original_path")) if p and Path(p).is_file()), None
                )
                artifacts.append(a | {"path": usable})
            else:
                # Serveur partagé : jamais de chemin du serveur ; le fichier se télécharge par un lien.
                stored = bool(a.get("stored_path")) and Path(a["stored_path"]).is_file()
                artifacts.append(a | {"stored_path": None, "path": None, "stored": stored})
        summary = self._summary(id_, question, meta)
        del summary["has_reasoning"], summary["artifact_names"], summary["tokens"]
        return summary | {
            "size_tokens": self._size(question, meta),
            "usage": self._usage(meta),
            "answer": meta.get("answer", ""),
            "reasoning": meta.get("reasoning", ""),
            "artifacts": artifacts,
            "created_by": meta.get("created_by", ""),
            "created_at": meta.get("created_at", ""),
            "revision": int(meta.get("revision", 1)),
            "last_used_at": meta.get("last_used_at") or None,
        }
