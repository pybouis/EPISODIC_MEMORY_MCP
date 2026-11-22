"""Tests de la recherche hybride : mots-clés des épisodes, glossaire de synonymes, bonus."""

from dataclasses import replace

import pytest

import episodic_memory.server as server
from episodic_memory.keywords import (
    Glossary,
    clean_keywords,
    content_tokens,
    coverage,
    haystack,
    match_keywords,
    normalize,
)

from test_security import ADMIN, ALICE, BOB, CHEF, as_user, put
from test_security import store as secured_store  # noqa: F401  (fixture)
from test_store import store  # noqa: F401  (fixture)

Q_PL = "Donne moi le P&L de 2023 par division en euro"
KW_PL = ["P&L", "PnL", "compte de résultat", "profit and loss"]


def best(store, question, **kwargs):  # noqa: F811
    return store.search(question, top_k=1, min_similarity=0.0, **kwargs)[0]


# ------------------------------------------------------------------- normalisation


def test_normalize_merges_spellings():
    assert normalize("P&L") == normalize("P & L") == normalize("PL") == "pl"
    assert normalize("Compte de Résultat") == "compte de resultat"
    assert normalize("Effets de change") == normalize("effet de change")
    assert normalize("ventes & marges") == "vente marge"  # « & » entre deux mots : pas de soudure


def test_clean_keywords_drops_duplicates_and_noise():
    assert clean_keywords([" P&L ", "p & l", "PL", "x", "", "compte  de|résultat"]) == ["P&L", "compte de résultat"]


def test_match_is_on_whole_terms():
    assert match_keywords(haystack("le PL SAFT 2022"), ["P&L"]) == ["P&L"]
    assert match_keywords(haystack("le plan SAFT 2022"), ["P&L"]) == []
    assert match_keywords(haystack("les comptes de résultat 2022"), ["compte de résultat"]) == ["compte de résultat"]


# ---------------------------------------------------------------- bonus de recherche


def test_keyword_in_question_adds_bonus(store):  # noqa: F811
    ep = store.add(Q_PL, "réponse", keywords=KW_PL)
    assert ep["keywords"] == KW_PL

    for question, expected in (("le PL SAFT 2022", ["P&L"]), ("le PnL SAFT 2022", ["PnL"])):
        m = best(store, question)
        assert m["matched_keywords"] == expected
        assert m["similarity"] == pytest.approx(min(1.0, m["vector_similarity"] + 0.2), abs=1e-4)

    m = best(store, "quel est l'effectif de SAFT en 2022")
    assert m["matched_keywords"] == [] and m["similarity"] == m["vector_similarity"]


def test_bonus_makes_the_episode_pass_the_threshold(store):  # noqa: F811
    store.add(Q_PL, "réponse", keywords=KW_PL)
    vector = best(store, "le PL SAFT 2022")["vector_similarity"]
    assert store.search("le PL SAFT 2022", min_similarity=vector + 0.1)  # passe grâce au bonus
    store.settings = replace(store.settings, keyword_bonus=0.0)
    assert store.search("le PL SAFT 2022", min_similarity=vector + 0.1) == []


def test_no_bonus_below_the_vector_floor(store):  # noqa: F811
    store.add(Q_PL, "réponse", keywords=KW_PL)
    store.settings = replace(store.settings, keyword_floor=0.99)
    m = best(store, "le PL SAFT 2022")
    assert m["matched_keywords"] == [] and m["similarity"] == m["vector_similarity"]


def test_keyword_match_outranks_a_closer_vector(store):  # noqa: F811
    other = store.add("Quel est l'effectif SAFT par pays ?", "réponse")
    ep = store.add(Q_PL, "réponse", keywords=KW_PL)
    # Vectoriellement, l'autre épisode (qui partage « SAFT ») est devant…
    ranked = store.search("PL SAFT", top_k=2, min_similarity=0.0)
    by_id = {m["id"]: m for m in ranked}
    assert by_id[other["id"]]["vector_similarity"] > by_id[ep["id"]]["vector_similarity"]
    # … mais le bonus du mot-clé fait remonter le bon, même avec top_k=1.
    store.settings = replace(store.settings, keyword_bonus=0.5)
    assert best(store, "PL SAFT")["id"] == ep["id"]


def test_query_side_keywords(store):  # noqa: F811
    store.add(Q_PL, "réponse", keywords=KW_PL)
    assert best(store, "le résultat de SAFT en 2022")["matched_keywords"] == []
    assert best(store, "le résultat de SAFT en 2022", keywords=["P&L"])["matched_keywords"] == ["P&L"]


# ------------------------------------------------------- mots-clés seuls (couverture)

Q_CASH = "Avec le cube REPORT de CXMD, analyser le cash par Unit en 2023 et donner des pistes d'optimisation."


def test_coverage_counts_meaningful_words_only():
    assert content_tokens("Donne-moi la trésorerie par Unit en 2022 ?") == ["tresorerie", "unit"]
    assert coverage("la trésorerie par Unit", Q_CASH, ["cash", "trésorerie"]) == 1.0
    assert coverage("la trésorerie de SAFT en 2022", Q_CASH, ["cash", "trésorerie"]) == 0.5
    assert coverage("et en 2022 ?", Q_CASH, ["cash"]) == 0.0  # aucun mot porteur de sens


def test_keyword_alone_matches_a_short_rephrased_question(store):  # noqa: F811
    ep = store.add(Q_CASH, "réponse", keywords=["cash", "trésorerie"])
    store.settings = replace(store.settings, keyword_floor=0.99)  # le bonus vectoriel est hors jeu

    m = store.search("la trésorerie par Unit", min_similarity=0.75)[0]  # tous les mots sont dans l'épisode
    assert m["id"] == ep["id"] and m["matched_keywords"] == ["trésorerie"]
    assert m["similarity"] == m["keyword_coverage"] == 1.0 and m["vector_similarity"] < 0.75

    # Un mot-clé concorde, mais la moitié de la question est étrangère à l'épisode : pas de rapprochement.
    m = best(store, "la trésorerie de SAFT")
    assert m["matched_keywords"] == [] and m["keyword_coverage"] is None and m["similarity"] == m["vector_similarity"]
    # Sans mot-clé concordant, la couverture seule ne compte pas.
    assert best(store, "les pistes d'optimisation par Unit")["matched_keywords"] == []

    store.settings = replace(store.settings, keyword_min_coverage=1.1)  # désactivé
    assert store.search("la trésorerie par Unit", min_similarity=0.75) == []


# ------------------------------------------------------------------------- glossaire


def test_glossary_links_synonyms_across_episodes(store):  # noqa: F811
    store.add("Analyser le BFR par Unit en 2023", "réponse", keywords=["BFR"])
    assert best(store, "évolution du working capital")["matched_keywords"] == []

    assert store.glossary.add(["BFR", "working capital"]) == ["BFR", "working capital"]
    assert best(store, "évolution du working capital")["matched_keywords"] == ["BFR"]
    # Un terme déjà connu fusionne les groupes.
    assert store.glossary.add(["Working Capital", "WCR"]) == ["Working Capital", "WCR", "BFR"]
    assert best(store, "le WCR 2022")["matched_keywords"] == ["BFR"]


def test_glossary_file_is_reread_and_tolerates_garbage(tmp_path):
    path = tmp_path / "glossary.json"
    glossary = Glossary(path)
    assert glossary.variants("P&L") == {"pl"}
    path.write_text('{"synonyms": [["P&L", "compte de résultat"]]}', encoding="utf-8")
    assert glossary.variants("P&L") == {"pl", "compte de resultat"}
    with pytest.raises(ValueError):
        glossary.add(["P&L", "p & l"])  # un seul terme distinct


# --------------------------------------------------------------- cycle de vie, droits


def test_keywords_survive_replace_export_and_can_be_edited(store):  # noqa: F811
    ep = store.add(Q_PL, "réponse", keywords=KW_PL)
    assert store.add(Q_PL, "v2", replace_episode_id=ep["id"])["keywords"] == KW_PL
    assert store.add(Q_PL, "v3", replace_episode_id=ep["id"], keywords=["P&L"])["keywords"] == ["P&L"]

    edited = store.set_keywords(ep["id"], ["P&L", "résultat"])
    assert edited["keywords"] == ["P&L", "résultat"] and edited["revision"] == 3
    with pytest.raises(KeyError):
        store.set_keywords("ep_absent", ["x"])

    path = store.export_json()
    store.delete(ep["id"])
    store.import_json(path)
    assert store.get(ep["id"])["keywords"] == ["P&L", "résultat"]


def test_tools_carry_keywords_and_check_rights(secured_store):  # noqa: F811
    secured_store.settings = replace(secured_store.settings, keyword_floor=0.0)  # embedding factice : cosinus bas
    ep = put(ALICE, Q_PL, keywords=KW_PL)
    found = as_user(ALICE, server.find_similar_question, "le PL SAFT 2022", min_similarity=0.0)
    assert found["matches"][0]["matched_keywords"] == ["P&L"]

    assert as_user(BOB, server.set_episode_keywords, ep["id"], ["x y"], True)["status"] == "forbidden"
    assert as_user(ALICE, server.set_episode_keywords, ep["id"], ["P&L"], False)["status"] == "refused"
    assert as_user(ALICE, server.set_episode_keywords, ep["id"], ["P&L", "marge"], True)["keywords"] == ["P&L", "marge"]

    assert as_user(ALICE, server.add_synonyms, ["P&L", "résultat"], True)["status"] == "forbidden"
    assert as_user(CHEF, server.add_synonyms, ["P&L", "résultat"], True)["synonyms"] == ["P&L", "résultat"]
    assert as_user(ADMIN, server.add_synonyms, ["seul"], True)["status"] == "error"
