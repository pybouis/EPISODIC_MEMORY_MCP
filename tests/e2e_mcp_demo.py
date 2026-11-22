"""Démonstration de bout en bout : lance le vrai serveur MCP (stdio, vrai modèle
d'embedding) sur une base temporaire et déroule le protocole en 4 étapes.

    python tests/e2e_mcp_demo.py

Ce n'est pas un test pytest : le premier lancement télécharge le modèle (~220 Mo).
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import Client, StdioServerParameters

TM1_QUESTION = "Comment créer un cube dans TM1 avec un processus TurboIntegrator ?"
TM1_ANSWER = "Processus TI livré dans creer_cube.xlsx : CubeCreate dans le Prolog, après test CubeExists."
TM1_REASONING = """\
**Contexte** : créer le cube Ventes sans écraser un cube existant.
**Sources** : serveur TM1 « prod », dimensions Produit, Période, Mesure.
**Démarche** : 1. lister les dimensions (}Dimensions) 2. CubeExists('Ventes') 3. CubeCreate dans le Prolog.
**Décisions** : Mesure en dernière dimension (convention maison).
**Pièges** : CubeCreate dans l'onglet Data s'exécute une fois par ligne de la source -> erreur.
**Reprise** : pour un autre cube, ne changer que le nom et la liste des dimensions."""

OTHERS = [
    ("Comment configurer OAuth pour un serveur MCP dans watsonx Orchestrate ?", "Créer une connexion OAuth2 puis…"),
    ("Quelle est la différence entre une liste et un tuple en Python ?", "Une liste est mutable, un tuple non."),
]

QUERIES = [
    "Comment est-ce que je crée un cube TM1 via TurboIntegrator ?",  # paraphrase
    "How do I create a TM1 cube from a TI process?",  # même question en anglais
    "Comment supprimer une dimension dans TM1 ?",  # même domaine, autre question
    "Quel temps fait-il à Bruxelles ?",  # sans rapport
]


def data(result):
    return result.structured_content or json.loads(result.content[0].text)


async def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        env = dict(
            os.environ,
            EPISODIC_DB_DIR=str(tmp / "chroma"),
            EPISODIC_ARTIFACTS_DIR=str(tmp / "artifacts"),
            EPISODIC_USER="demo@example.com",
            PYTHONIOENCODING="utf-8",
        )
        params = StdioServerParameters(command=sys.executable, args=["-m", "episodic_memory"], env=env)
        async with Client(params, read_timeout_seconds=600) as client:

            async def call(tool, **args):
                return data(await client.call_tool(tool, args))

            print("Outils :", [t.name for t in (await client.list_tools()).tools])
            print("Instructions transmises au bot :", bool(client.instructions))

            r = await call("store_validated_episode", question="q", answer="a", reasoning="r", confirmed_by_user=False)
            print("Stockage sans validation ->", r["status"])
            r = await call("store_validated_episode", question="q", answer="a", reasoning="x" * 5000, confirmed_by_user=True)
            print("Raisonnement de 5000 caractères ->", r["status"], f"(max {r['max_chars']})")

            # Étape 4 : épisode complet = réponse + raisonnement synthétisé + fichier produit.
            produced = tmp / "creer_cube.xlsx"
            produced.write_bytes(b"faux classeur Excel")
            r = await call(
                "store_validated_episode",
                question=TM1_QUESTION,
                answer=TM1_ANSWER,
                reasoning=TM1_REASONING,
                artifacts=[{"path": str(produced), "description": "Script TI commenté"}],
                confirmed_by_user=True,
            )
            ep = r["episode"]
            print("Stocké :", r["status"], ep["id"], ep["validated_by"], ep["validated_at"])
            produced.unlink()  # l'original disparaît : la copie d'archive doit suffire
            for q, a in OTHERS:
                await call("store_validated_episode", question=q, answer=a, reasoning="Réponse directe.", confirmed_by_user=True)

            # Similarités brutes (seuil 0) pour juger du seuil par défaut.
            for query in QUERIES:
                best = (await call("find_similar_question", question=query, min_similarity=0.0, top_k=1))["matches"][0]
                print(f"\n« {query} »\n   meilleur : {best['similarity']:.3f}  « {best['question']} »")
                r = await call("find_similar_question", question=query)
                print(f"   proposé au seuil {r['threshold']} : {[m['id'] for m in r['matches']] or 'rien'}")

            # Étapes 1 à 3 sur la paraphrase.
            match = (await call("find_similar_question", question=QUERIES[0]))["matches"][0]
            assert "answer" not in match and "reasoning" not in match
            print("\nProposition :", match["question"], "| raisonnement :", match["has_reasoning"], "| fichiers :", match["artifact_names"])
            ep = (await call("get_validated_episode", episode_id=match["id"]))["episode"]
            print("Après validation utilisateur ->", ep["answer"], "| utilisations :", ep["use_count"])
            print(f"Raisonnement restitué ({len(ep['reasoning'])} caractères) :\n{ep['reasoning']}")
            art = ep["artifacts"][0]
            print("Fichier archivé lisible :", Path(art["path"]).read_bytes() == b"faux classeur Excel", "|", art["description"])

            # Doublon détecté, puis remplacement qui conserve raisonnement et fichiers.
            r = await call("store_validated_episode", question=TM1_QUESTION, answer="v2", reasoning="r", confirmed_by_user=True)
            print("\nRe-stockage de la même question ->", r["status"])
            r = await call(
                "store_validated_episode", question=TM1_QUESTION, answer="v2", reasoning=TM1_REASONING,
                confirmed_by_user=True, replace_episode_id=match["id"],
            )
            print("Remplacement ->", r["status"], "révision", r["episode"]["revision"], "| fichiers conservés :", len(r["episode"]["artifacts"]))
            print("Total :", (await call("list_episodes"))["total"])


asyncio.run(main())
