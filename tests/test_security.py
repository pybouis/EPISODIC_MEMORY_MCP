"""Tests de la couche de sécurité : rôles, périmètre par groupe, annuaire local, mapping PA."""

import asyncio
from dataclasses import replace

import pytest

import episodic_memory.server as server
from episodic_memory.config import load_settings
from episodic_memory.security import (
    AccessDenied,
    GroupRequired,
    LocalProvider,
    PlanningAnalyticsProvider,
    Principal,
    UserDirectory,
    principal_from_pa,
)
from episodic_memory.store import EpisodicStore

from test_store import FakeEmbedder

ADMIN = Principal("root@x.com", is_admin=True)
CHEF = Principal("chef@x.com", admin_groups=frozenset({"finance"}))
ALICE = Principal("alice@x.com", groups=frozenset({"finance"}))
BOB = Principal("bob@x.com", groups=frozenset({"finance"}))
CAROL = Principal("carol@x.com", groups=frozenset({"rh"}))
MULTI = Principal("multi@x.com", groups=frozenset({"finance", "rh"}))


@pytest.fixture
def store(tmp_path, monkeypatch):
    settings = replace(
        load_settings({}),
        db_dir=tmp_path / "chroma",
        export_dir=tmp_path / "exports",
        artifacts_dir=tmp_path / "artifacts",
        user="local",
        threshold=0.6,
        transport="http",
        auth_provider="local",
        archive_artifacts=False,
        glossary_path=tmp_path / "glossary.json",
    )
    st = EpisodicStore(settings, embedder=FakeEmbedder())
    monkeypatch.setattr(server, "_store", st)
    monkeypatch.setattr(server, "_directory", None)
    return st


def as_user(principal, fn, *args, **kwargs):
    token = server._current.set(principal)
    try:
        return fn(*args, **kwargs)
    finally:
        server._current.reset(token)


def put(principal, question, **kwargs):
    res = as_user(principal, server.store_validated_episode, question, "réponse", "raisonnement", True, **kwargs)
    assert res["status"] in ("stored", "replaced"), res
    return res["episode"]


Q_FIN = "Comment calculer la marge brute du budget finance ?"
Q_RH = "Combien de jours de congés pour un nouvel embauché ?"


# ---------------------------------------------------------------- périmètre de lecture


def test_unauthenticated_http_call_is_forbidden(store):
    assert server.list_episodes()["status"] == "forbidden"


def test_search_list_and_get_are_limited_to_own_groups(store):
    fin = put(ALICE, Q_FIN)
    rh = put(CAROL, Q_RH)
    assert fin["group"] == "FINANCE" and fin["validated_by"] == "alice@x.com"

    found = as_user(ALICE, server.find_similar_question, Q_RH, min_similarity=0.0)
    assert [m["id"] for m in found["matches"]] == [fin["id"]] and found["episodes_in_base"] == 1
    assert [e["id"] for e in as_user(ALICE, server.list_episodes)["episodes"]] == [fin["id"]]
    # L'épisode d'un autre groupe est « introuvable » : son existence n'est pas révélée.
    assert as_user(ALICE, server.get_validated_episode, rh["id"])["status"] == "not_found"
    assert as_user(ALICE, server.get_validated_episode, fin["id"])["status"] == "ok"

    assert as_user(ADMIN, server.list_episodes)["total"] == 2
    assert as_user(MULTI, server.list_episodes, group="rh")["total"] == 1
    assert as_user(ALICE, server.list_episodes, group="rh")["total"] == 0


def test_duplicate_detection_does_not_leak_across_groups(store):
    put(ALICE, Q_FIN)
    # La même question dans un autre groupe n'est pas signalée comme doublon.
    assert put(CAROL, Q_FIN)["group"] == "RH"
    assert as_user(BOB, server.store_validated_episode, Q_FIN, "r", "x", True)["status"] == "duplicate_found"


# ----------------------------------------------------------------------- stockage


def test_group_is_required_when_user_has_several(store):
    res = as_user(MULTI, server.store_validated_episode, Q_FIN, "r", "x", True)
    assert res["status"] == "group_required" and res["groups"] == ["FINANCE", "RH"]
    assert put(MULTI, Q_FIN, group="rh")["group"] == "RH"


def test_cannot_store_in_foreign_group(store):
    assert as_user(ALICE, server.store_validated_episode, Q_RH, "r", "x", True, group="rh")["status"] == "forbidden"
    assert put(ADMIN, Q_RH, group="rh")["group"] == "RH"
    assert put(ADMIN, "Question sans groupe précisé ?")["group"] == "ADMIN"


def test_replace_rules(store):
    ep = put(ALICE, Q_FIN)
    denied = as_user(BOB, server.store_validated_episode, Q_FIN, "v2", "x", True, replace_episode_id=ep["id"])
    assert denied["status"] == "forbidden"
    assert put(ALICE, Q_FIN, replace_episode_id=ep["id"])["revision"] == 2
    revalidated = put(CHEF, Q_FIN, replace_episode_id=ep["id"])
    assert revalidated["validated_by"] == "chef@x.com" and revalidated["created_by"] == "alice@x.com"
    assert revalidated["group"] == "FINANCE"


# -------------------------------------------------------------------- suppression


def test_author_deletes_only_what_he_created_and_validated(store):
    ep = put(ALICE, Q_FIN)
    assert as_user(BOB, server.delete_episode, ep["id"], True)["status"] == "forbidden"
    assert as_user(CAROL, server.delete_episode, ep["id"], True)["status"] == "not_found"
    assert as_user(ALICE, server.delete_episode, ep["id"], True)["status"] == "deleted"


def test_author_loses_delete_right_once_revalidated_by_group_admin(store):
    ep = put(ALICE, Q_FIN)
    put(CHEF, Q_FIN, replace_episode_id=ep["id"])
    assert as_user(ALICE, server.delete_episode, ep["id"], True)["status"] == "forbidden"
    assert as_user(CHEF, server.delete_episode, ep["id"], True)["status"] == "deleted"


def test_group_admin_is_limited_to_his_group(store):
    rh = put(CAROL, Q_RH)
    assert as_user(CHEF, server.delete_episode, rh["id"], True)["status"] == "not_found"
    assert as_user(ADMIN, server.delete_episode, rh["id"], True)["status"] == "deleted"


def test_export_scope(store):
    put(ALICE, Q_FIN)
    put(CAROL, Q_RH)
    assert as_user(ALICE, server.export_episodes)["status"] == "forbidden"
    assert as_user(CHEF, server.export_episodes)["episodes"] == 1
    assert as_user(ADMIN, server.export_episodes)["episodes"] == 2


def test_shared_server_never_opens_artifact_paths(store, tmp_path):
    secret = tmp_path / "secret.env"
    secret.write_text("x")
    res = as_user(ALICE, server.store_validated_episode, Q_FIN, "r", "x", True,
                  artifacts=[server.ArtifactRef(path=str(secret), description="d")])
    art = res["episode"]["artifacts"][0]
    # Le serveur ne lit pas le chemin fourni et ne renvoie jamais de chemin : il attend l'envoi.
    assert art["stored_path"] is None and art["path"] is None and art["stored"] is False
    assert art["original_path"] == str(secret) and not store.settings.artifacts_dir.exists()
    assert [u["name"] for u in res["uploads"]] == ["secret.env"] and "curl" in res["uploads"][0]["command"]


def test_files_are_uploaded_then_shared_with_the_group(store, tmp_path):
    from starlette.testclient import TestClient

    client = TestClient(server.AuthMiddleware(server.mcp.streamable_http_app(stateless_http=True), provider=object()))
    path = lambda url: "/files/" + url.split("/files/")[1]

    res = as_user(ALICE, server.store_validated_episode, Q_FIN, "r", "x", True, artifacts=[
        server.ArtifactRef(path=r"C:\poste\a\rapport.xlsx"), server.ArtifactRef(path="/home/alice/b/rapport.xlsx"),
    ])
    ep_id = res["episode"]["id"]
    up1, up2 = res["uploads"]
    assert [up1["name"], up2["name"]] == ["rapport.xlsx", "rapport_1.xlsx"]

    assert client.put(path(up1["upload_url"]), content=b"").status_code == 400
    sent = client.put(path(up1["upload_url"]), content=b"contenu excel")
    assert sent.status_code == 200 and sent.json() == {"status": "stored", "name": "rapport.xlsx", "size_bytes": 13}
    assert client.put(path(up1["upload_url"]), content=b"rejoue").status_code == 404  # usage unique
    assert client.put("/files/up/inconnu", content=b"x").status_code == 404
    assert client.get(path(up2["upload_url"]).replace("/up/", "/dl/")).status_code == 404  # pas le bon usage
    assert not (store.settings.artifacts_dir / ".incoming").exists() or not any((store.settings.artifacts_dir / ".incoming").iterdir())

    # Un collègue du même groupe obtient un lien de téléchargement ; un autre groupe, rien.
    got = as_user(BOB, server.get_validated_episode, ep_id)["episode"]["artifacts"]
    assert got[0]["stored"] and "download_url" not in got[1] and got[0]["path"] is None
    assert str(store.settings.artifacts_dir) not in str(got)
    resp = client.get(path(got[0]["download_url"]))
    assert resp.status_code == 200 and resp.content == b"contenu excel"
    assert "rapport.xlsx" in resp.headers["content-disposition"]
    assert as_user(CAROL, server.get_validated_episode, ep_id)["status"] == "not_found"

    # Fichier trop volumineux : refusé, rien n'est gardé.
    big = replace(store.settings, artifact_max_bytes=4)
    object.__setattr__(store, "settings", big)
    assert client.put(path(up2["upload_url"]), content=b"12345").status_code == 413
    object.__setattr__(store, "settings", replace(big, artifact_max_bytes=10**6))

    # Un remplacement sans `artifacts` conserve les fichiers ; la suppression les efface.
    kept = as_user(ALICE, server.store_validated_episode, Q_FIN, "v2", "x", True, replace_episode_id=ep_id)
    assert kept["episode"]["artifacts"][0]["stored"] and [u["name"] for u in kept["uploads"]] == ["rapport_1.xlsx"]
    assert as_user(ALICE, server.delete_episode, ep_id, True)["status"] == "deleted"
    assert not (store.settings.artifacts_dir / ep_id).exists()
    assert client.get(path(got[0]["download_url"])).status_code == 404


def test_expired_file_link_is_refused(store, monkeypatch):
    url = server._file_link("dl", "ep_x", "a.txt", "alice")
    ticket = url.rsplit("/", 1)[1]
    assert server._ticket("dl", ticket) and server._ticket("up", ticket) is None
    monkeypatch.setattr(server.time, "time", lambda: 10**12)
    assert server._ticket("dl", ticket) is None


# ------------------------------------------------------------------------ migration


def test_legacy_episodes_are_admin_only_until_assigned(store):
    ep = store.add(Q_FIN, "r")
    store._col.update(ids=[ep["id"]], metadatas=[{k: v for k, v in store._col.get(ids=[ep["id"]])["metadatas"][0].items() if k != "group"} | {"group": ""}])
    assert as_user(ALICE, server.list_episodes)["total"] == 0
    assert store.assign_group("admin") == 1 and store.assign_group("admin") == 0
    assert store.get(ep["id"])["group"] == "ADMIN"


# ------------------------------------------------------------------- annuaire local


def test_local_directory_tokens_and_roles(store):
    directory = UserDirectory(store.client)
    directory.save("Alice@x.com", groups=["finance"])
    token = directory.new_token("alice@x.com")
    provider = LocalProvider(directory)

    who = asyncio.run(provider.authenticate(token))
    assert who.user == "Alice@x.com" and who.groups == {"FINANCE"} and not who.is_admin
    assert asyncio.run(provider.authenticate("emk_faux")) is None
    assert token not in str(directory._col.get(include=["metadatas"]))  # seul le hash est stocké

    new = directory.new_token("alice@x.com")
    assert asyncio.run(provider.authenticate(token)) is None  # l'ancien jeton est révoqué
    directory.save("alice@x.com", active=False)
    assert asyncio.run(provider.authenticate(new)) is None
    assert store.count() == 0  # l'annuaire ne pollue pas la collection des épisodes


def test_user_management_delegation(store):
    created = as_user(ADMIN, server.save_user, "chef@x.com", True, admin_of_groups=["finance"])
    assert created["status"] == "created" and created["token"].startswith("emk_")
    assert created["user"]["groups"] == ["FINANCE"]

    assert as_user(ALICE, server.save_user, "x@x.com", True, groups=["finance"])["status"] == "forbidden"
    assert as_user(CHEF, server.save_user, "x@x.com", True, groups=["rh"])["status"] == "forbidden"
    assert as_user(CHEF, server.save_user, "x@x.com", True, groups=["finance"], is_admin=True)["status"] == "forbidden"
    assert as_user(CHEF, server.save_user, "bob@x.com", True, groups=["finance"])["status"] == "created"

    as_user(ADMIN, server.save_user, "multi@x.com", True, groups=["finance", "rh"])
    # Le chef de FINANCE ne peut ni retirer RH, ni désactiver, ni renouveler le jeton d'un membre d'un autre groupe.
    assert as_user(CHEF, server.save_user, "multi@x.com", True, groups=["finance"])["status"] == "forbidden"
    assert as_user(CHEF, server.save_user, "multi@x.com", True, active=False)["status"] == "forbidden"
    assert as_user(CHEF, server.reset_user_token, "multi@x.com", True)["status"] == "forbidden"
    assert as_user(CHEF, server.reset_user_token, "bob@x.com", True)["status"] == "token_reset"
    assert as_user(MULTI, server.reset_user_token, "multi@x.com", True)["status"] == "token_reset"

    assert {u["user"] for u in as_user(CHEF, server.list_users)["users"]} == {"chef@x.com", "bob@x.com", "multi@x.com"}
    assert as_user(ALICE, server.list_users)["status"] == "forbidden"


# -------------------------------------------------------------- Planning Analytics


def pa_settings(**kwargs):
    return replace(load_settings({}), pa_url="https://pa.example/api/t/v0/tm1/Base", **kwargs)


def test_pa_groups_mapping_with_prefix():
    who = principal_from_pa(
        "pierre@x.com", ["ADMIN", "particuliers", "EPISODIC_Finance", "Episodic_RH_Admin", "}tp_Everyone"], pa_settings()
    )
    assert not who.is_admin  # le groupe ADMIN de PA ne donne rien sans configuration explicite
    assert who.groups == {"FINANCE", "RH"} and who.admin_groups == {"RH"}
    assert principal_from_pa("a", ["episodic_admin"], pa_settings()).is_admin


def test_pa_groups_mapping_without_prefix():
    settings = pa_settings(pa_group_prefix="", pa_admin_groups=("ADMIN",))
    who = principal_from_pa("pierre@x.com", ["particuliers", "pme/pmi", "}tp_Everyone"], settings)
    assert who.groups == {"PARTICULIERS", "PME/PMI"} and not who.is_admin
    assert principal_from_pa("root", ["ADMIN", "cdg"], settings).is_admin


def test_pa_provider_caches_and_rejects():
    calls = []

    async def fetch(token):
        calls.append(token)
        if token != "bon":
            return None
        return {"Name": "id42", "FriendlyName": "pierre@x.com", "Groups": [{"Name": "EPISODIC_FINANCE"}]}

    provider = PlanningAnalyticsProvider(pa_settings(), fetch=fetch)
    who = asyncio.run(provider.authenticate("bon"))
    assert who.user == "pierre@x.com" and who.groups == {"FINANCE"}
    asyncio.run(provider.authenticate("bon"))
    assert calls == ["bon"]  # second appel servi par le cache
    assert asyncio.run(provider.authenticate("mauvais")) is None
    assert asyncio.run(provider.authenticate("")) is None

    with pytest.raises(ValueError):
        PlanningAnalyticsProvider(replace(load_settings({}), pa_url=""))


def test_principal_rules():
    with pytest.raises(GroupRequired):
        MULTI.resolve_group(None, "ADMIN")
    with pytest.raises(AccessDenied):
        Principal("x").resolve_group(None, "ADMIN")
    with pytest.raises(AccessDenied):
        ALICE.export_scope()
    assert CHEF.groups == {"FINANCE"} and CHEF.describe()["role"] == "group_admin"
