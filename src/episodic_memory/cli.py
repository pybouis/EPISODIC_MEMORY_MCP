"""Ligne de commande : consulter, sauvegarder et administrer la base hors du chat.

    episodic-memory serve                               (lance le serveur MCP)
    episodic-memory list [--group G]
    episodic-memory search "ma question" [--keywords "P&L,compte de résultat"]
    episodic-memory set-keywords ep_xxxxxxxxxxxx "P&L,PnL,compte de résultat"
    episodic-memory glossary list
    episodic-memory glossary add "BFR,besoin en fonds de roulement,working capital"
    episodic-memory show ep_xxxxxxxxxxxx
    episodic-memory export [chemin.json]
    episodic-memory import chemin.json
    episodic-memory delete ep_xxxxxxxxxxxx
    episodic-memory assign-group ADMIN [--all]          (rattache les épisodes sans groupe)
    episodic-memory tokens [--group G]                  (tokens par épisode et cumul)
    episodic-memory backfill-usage [--dry-run] [--force] [--any-window]  (conso des épisodes déjà stockés)
    episodic-memory user list
    episodic-memory user add NOM [--groups A,B] [--admin-of A] [--admin]
    episodic-memory user set NOM [--groups A,B] [--admin-of A] [--admin|--no-admin] [--enable|--disable]
    episodic-memory user token NOM                      (nouveau jeton, affiché une seule fois)
    episodic-memory user remove NOM

La ligne de commande agit en administrateur : elle suppose un accès au serveur lui-même.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings


def _csv(value: str | None) -> list[str] | None:
    return None if value is None else [v.strip() for v in value.split(",") if v.strip()]


def _user_line(u: dict) -> str:
    role = "admin" if u["is_admin"] else "admin de " + ",".join(u["admin_of_groups"]) if u["admin_of_groups"] else "user"
    state = "" if u["active"] else "  [désactivé]"
    return f"{u['user']}  ({role})  groupes : {', '.join(u['groups']) or '-'}{state}"


def _tok(tokens: dict) -> str:
    if tokens["consumed"] is None:
        return f"taille {tokens['size']} / conso -"
    return f"taille {tokens['size']} / conso {_n(tokens['consumed'])} (sortie {_n(tokens['output'])})"


def _n(value: int | None) -> str:
    return "-" if value is None else f"{value:,}".replace(",", " ")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="episodic-memory", description="Base de référence de la mémoire épisodique")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="lancer le serveur MCP (stdio ou http selon EPISODIC_TRANSPORT)")
    p = sub.add_parser("list", help="lister les épisodes")
    p.add_argument("--group")
    p = sub.add_parser("search", help="chercher les questions proches")
    p.add_argument("question")
    p.add_argument("--min-similarity", type=float, default=0.0, help="seuil (défaut 0 : tout afficher)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--keywords", help="autres formes des termes de la question, séparées par des virgules")
    p = sub.add_parser("set-keywords", help="remplacer les mots-clés d'un épisode")
    p.add_argument("id")
    p.add_argument("keywords", help="liste complète, séparée par des virgules (\"\" pour tout retirer)")
    glossary = sub.add_parser("glossary", help="synonymes partagés de la recherche par mots-clés")
    gsub = glossary.add_subparsers(dest="gcmd", required=True)
    gsub.add_parser("list", help="afficher les groupes de synonymes")
    p = gsub.add_parser("add", help="déclarer des termes équivalents")
    p.add_argument("terms", help="termes séparés par des virgules")
    p = sub.add_parser("show", help="afficher un épisode complet")
    p.add_argument("id")
    p = sub.add_parser("export", help="exporter la base en JSON")
    p.add_argument("path", nargs="?", type=Path)
    p = sub.add_parser("import", help="importer un export JSON")
    p.add_argument("path", type=Path)
    p = sub.add_parser("delete", help="supprimer un épisode")
    p.add_argument("id")
    p = sub.add_parser("assign-group", help="rattacher à un groupe les épisodes qui n'en ont pas")
    p.add_argument("group")
    p.add_argument("--all", action="store_true", help="rattacher TOUS les épisodes, même ceux qui ont déjà un groupe")

    p = sub.add_parser("tokens", help="tokens par épisode : taille stockée et consommation mesurée")
    p.add_argument("--group")
    p = sub.add_parser("backfill-usage", help="retrouver dans les journaux Claude Code la consommation des épisodes existants")
    p.add_argument("--force", action="store_true", help="remesurer aussi les épisodes déjà mesurés")
    p.add_argument("--dry-run", action="store_true", help="afficher sans enregistrer")
    p.add_argument(
        "--any-window",
        action="store_true",
        help="accepter aussi les mesures sans recherche préalable (fenêtre moins fiable : toute la session "
        "ou depuis le stockage précédent)",
    )

    user = sub.add_parser("user", help="gérer les utilisateurs (EPISODIC_AUTH_PROVIDER=local)")
    usub = user.add_subparsers(dest="ucmd", required=True)
    usub.add_parser("list", help="lister les utilisateurs")
    for name, text in (("add", "créer un utilisateur et afficher son jeton"), ("set", "modifier un utilisateur")):
        p = usub.add_parser(name, help=text)
        p.add_argument("name")
        p.add_argument("--groups", help="groupes dont il est membre, séparés par des virgules")
        p.add_argument("--admin-of", help="groupes dont il est l'administrateur")
        p.add_argument("--admin", dest="is_admin", action="store_const", const=True, help="administrateur de la base")
        if name == "set":
            p.add_argument("--no-admin", dest="is_admin", action="store_const", const=False)
            p.add_argument("--enable", dest="active", action="store_const", const=True)
            p.add_argument("--disable", dest="active", action="store_const", const=False)
    p = usub.add_parser("token", help="générer un nouveau jeton")
    p.add_argument("name")
    p = usub.add_parser("remove", help="supprimer un utilisateur")
    p.add_argument("name")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.cmd == "serve":
        from .server import main as serve

        serve()
        return

    from .store import EpisodicStore

    store = EpisodicStore(load_settings())

    if args.cmd == "list":
        groups = [args.group.strip().upper()] if args.group else None
        for e in store.list(limit=100000, groups=groups):
            print(
                f"{e['id']}  {e['validated_at']}  [{e['group'] or 'sans groupe'}]  {e['validated_by']}  "
                f"(x{e['use_count']})  [{_tok(e['tokens'])}]  {e['question']}"
                + (f"  {{{', '.join(e['keywords'])}}}" if e["keywords"] else "")
            )
        print(f"{store.count(groups)} épisode(s) — base : {store.settings.db_dir}")
    elif args.cmd == "search":
        for m in store.search(
            args.question, top_k=args.top_k, min_similarity=args.min_similarity, keywords=_csv(args.keywords)
        ):
            hit = (
                f"  (vecteur {m['vector_similarity']:.3f}, couverture {m['keyword_coverage'] or 0:.2f}, "
                f"mots-clés : {', '.join(m['matched_keywords'])})"
            )
            print(f"{m['similarity']:.3f}  {m['id']}  [{m['group']}]  {m['question']}{hit if m['matched_keywords'] else ''}")
    elif args.cmd == "set-keywords":
        try:
            print("mots-clés : " + (", ".join(store.set_keywords(args.id, _csv(args.keywords))["keywords"]) or "(aucun)"))
        except KeyError as exc:
            sys.exit(str(exc.args[0]))
    elif args.cmd == "glossary":
        if args.gcmd == "add":
            try:
                print(" = ".join(store.glossary.add(_csv(args.terms))))
            except ValueError as exc:
                sys.exit(str(exc))
        else:
            for group in store.glossary.groups():
                print(" = ".join(group))
            print(f"glossaire : {store.glossary.path}")
    elif args.cmd == "show":
        print(json.dumps(store.get(args.id), ensure_ascii=False, indent=2))
    elif args.cmd == "export":
        print(store.export_json(args.path))
    elif args.cmd == "import":
        print(f"{store.import_json(args.path)} épisode(s) importé(s)")
    elif args.cmd == "delete":
        print("supprimé" if store.delete(args.id) else "introuvable")
    elif args.cmd == "assign-group":
        n = store.assign_group(args.group, only_missing=not args.all)
        print(f"{n} épisode(s) rattaché(s) au groupe {args.group.strip().upper()}")
    elif args.cmd == "tokens":
        groups = [args.group.strip().upper()] if args.group else None
        for e in store.list(limit=100000, groups=groups):
            print(f"{e['id']}  {_tok(e['tokens']):>50}  (x{e['use_count']})  {e['question'][:90]}")
        t = store.token_totals(groups)
        print(
            f"{t['episodes']} épisode(s) — taille stockée ≈ {_n(t['size_tokens'])} tokens ; "
            f"consommation mesurée sur {t['measured_episodes']} épisode(s) : {_n(t['consumed_tokens'])} tokens "
            f"(dont {_n(t['cache_read_tokens'])} relus du cache, {_n(t['output_tokens'])} en sortie)"
        )
    elif args.cmd == "backfill-usage":
        from .usage import scan_store_calls, transcript_usage

        by_question: dict[str, list[dict]] = {}
        for e in store.list(limit=100000):
            by_question.setdefault(e["question"].strip(), []).append(e)

        def similarity(q: str, candidates: list[str]) -> list[float]:
            vectors = store.embedder.embed([q, *candidates])
            return [sum(a * b for a, b in zip(vectors[0], v)) for v in vectors[1:]]

        best: dict[str, dict] = {}
        for path, question in scan_store_calls(store.settings.claude_projects_dir):
            for e in by_question.get(question, []):
                if e["tokens"]["consumed"] is not None and not args.force:
                    continue
                usage = transcript_usage(path, question, similarity)
                if usage and usage["window"] != "question" and not args.any_window:
                    continue
                # Plusieurs conversations ont pu stocker la même question : garder la plus récente.
                if usage and usage["ended_at"] and (e["id"] not in best or usage["ended_at"] > best[e["id"]]["ended_at"]):
                    best[e["id"]] = usage
        for episode_id, usage in best.items():
            if not args.dry_run:
                store.set_usage(episode_id, usage)
            print(f"{episode_id}  {usage['total_tokens']:>12,} tokens  (sortie {usage['output_tokens']:,}, cache lu {usage['cache_read_input_tokens']:,}, "
                f"{usage['api_calls']} appels, fenêtre {usage['window']})")
        print(f"{len(best)} épisode(s) {'mesurable(s)' if args.dry_run else 'mis à jour'}")
    elif args.cmd == "user":
        from .security import UserDirectory

        directory = UserDirectory(store.client)
        if args.ucmd == "list":
            for u in directory.list():
                print(_user_line(u))
            print(f"{directory.count()} utilisateur(s)")
        elif args.ucmd in ("add", "set"):
            exists = directory.get(args.name) is not None
            if args.ucmd == "add" and exists:
                sys.exit(f"L'utilisateur {args.name} existe déjà (utilisez « user set »).")
            if args.ucmd == "set" and not exists:
                sys.exit(f"Utilisateur inconnu : {args.name}")
            record = directory.save(
                args.name,
                is_admin=args.is_admin,
                groups=_csv(args.groups),
                admin_of_groups=_csv(args.admin_of),
                active=getattr(args, "active", None),
                created_by="cli",
            )
            print(_user_line(record))
            if args.ucmd == "add":
                print(f"Jeton (affiché une seule fois) : {directory.new_token(record['user'])}")
        elif args.ucmd == "token":
            try:
                print(f"Nouveau jeton (affiché une seule fois) : {directory.new_token(args.name)}")
            except KeyError as exc:
                sys.exit(str(exc.args[0]))
        elif args.ucmd == "remove":
            print("supprimé" if directory.delete(args.name) else "introuvable")


if __name__ == "__main__":
    main()
