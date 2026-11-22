"""Démonstration de bout en bout du serveur partagé : vrai serveur HTTP, vrais jetons.

    python tests/e2e_http_security.py [commande du serveur…]

Sans argument, lance `python -m episodic_memory.cli` ; on peut passer le chemin de l'exe
pour tester la version livrée :  python tests/e2e_http_security.py C:\\...\\episodic-memory-mcp.exe

Base temporaire, annuaire local. Ce n'est pas un test pytest (vrai modèle d'embedding).
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

PORT = 8017
URL = f"http://127.0.0.1:{PORT}"
Q = "Comment calculer la marge brute du budget finance ?"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    base_cmd = sys.argv[1:] or [sys.executable, "-m", "episodic_memory.cli"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        env = dict(
            os.environ,
            EPISODIC_ENV_FILE=str(tmp / "absent.env"),
            EPISODIC_DB_DIR=str(tmp / "chroma"),
            EPISODIC_ARTIFACTS_DIR=str(tmp / "artifacts"),
            EPISODIC_EXPORT_DIR=str(tmp / "exports"),
            EPISODIC_LOG_DIR=str(tmp / "logs"),
            EPISODIC_TRANSPORT="http",
            EPISODIC_AUTH_PROVIDER="local",
            MCP_HOST="127.0.0.1",
            MCP_PORT=str(PORT),
            MCP_PUBLIC_BASE_URL="https://demo-8017.euw.devtunnels.ms",
            PYTHONIOENCODING="utf-8",
        )

        def cli(*args: str) -> str:
            out = subprocess.run([*base_cmd, *args], env=env, capture_output=True, text=True, encoding="utf-8")
            if out.returncode:
                raise SystemExit(f"{args} -> {out.stdout}\n{out.stderr}")
            return out.stdout

        def add_user(name: str, *options: str) -> str:
            return re.search(r"emk_\S+", cli("user", "add", name, *options)).group(0)

        tokens = {
            "admin": add_user("root@x.com", "--admin"),
            "chef": add_user("chef@x.com", "--admin-of", "finance"),
            "alice": add_user("alice@x.com", "--groups", "finance"),
            "carol": add_user("carol@x.com", "--groups", "rh"),
        }
        print(cli("user", "list"))

        server = subprocess.Popen([*base_cmd, "serve"], env=env)
        try:
            for _ in range(120):
                try:
                    if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(1)
            else:
                raise SystemExit("le serveur n'a pas démarré")
            print("health :", httpx.get(f"{URL}/health").json())

            def call(who: str | None, tool: str, host: str | None = None, **arguments):
                headers = {"Accept": "application/json, text/event-stream"}
                if who:
                    headers["Authorization"] = f"Bearer {tokens.get(who, who)}"
                if host:
                    headers["Host"] = host
                body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
                resp = httpx.post(f"{URL}/mcp", json=body, headers=headers, timeout=600)
                if resp.status_code != 200:
                    return {"http": resp.status_code}
                result = resp.json()["result"]
                return result.get("structuredContent") or json.loads(result["content"][0]["text"])

            checks = []

            def check(label: str, ok: bool, detail="") -> None:
                checks.append(ok)
                print(("OK   " if ok else "ÉCHEC") + f"  {label}  {detail}")

            check("sans jeton -> 401", call(None, "whoami") == {"http": 401})
            check("faux jeton -> 401", call("emk_faux", "whoami") == {"http": 401})
            who = call("alice", "whoami")
            check("whoami alice", who.get("user") == "alice@x.com" and who.get("groups") == ["FINANCE"], who)
            check("via le tunnel (Host public)", call("alice", "whoami", host="demo-8017.euw.devtunnels.ms").get("role") == "user")
            check("Host inconnu refusé", "http" in call("alice", "whoami", host="evil.example"))

            stored = call("alice", "store_validated_episode", question=Q, answer="CA - coût des ventes.",
                          reasoning="**Démarche** : cube Budget.", confirmed_by_user=True)
            ep = stored.get("episode", {})
            check("alice stocke dans FINANCE", stored.get("status") == "stored" and ep.get("group") == "FINANCE"
                  and ep.get("validated_by") == "alice@x.com", stored.get("status"))
            check("alice retrouve sa question", len(call("alice", "find_similar_question", question=Q)["matches"]) == 1)
            check("carol (RH) ne voit rien", call("carol", "find_similar_question", question=Q) ==
                  {"matches": [], "threshold": 0.75, "episodes_in_base": 0})
            check("carol : épisode introuvable", call("carol", "get_validated_episode", episode_id=ep["id"])["status"] == "not_found")
            check("carol ne peut pas supprimer", call("carol", "delete_episode", episode_id=ep["id"], confirmed_by_user=True)["status"] == "not_found")
            check("alice ne peut pas exporter", call("alice", "export_episodes")["status"] == "forbidden")
            check("admin voit tout", call("admin", "list_episodes")["total"] == 1)
            # Fichier produit : envoyé au serveur par la commande curl renvoyée, puis partagé avec le groupe.
            report = tmp / "poste_alice" / "rapport marge.xlsx"
            report.parent.mkdir()
            report.write_bytes(b"contenu du classeur " * 1000)
            with_file = call("alice", "store_validated_episode", question="Où est le rapport de marge du budget ?",
                             answer="Dans le classeur joint.", reasoning="x", confirmed_by_user=True,
                             artifacts=[{"path": str(report), "description": "Rapport de marge"}])
            uploads = with_file.get("uploads", [])
            check("le serveur demande l'envoi du fichier", len(uploads) == 1 and uploads[0]["upload_url"].startswith("https://demo-8017"))
            local = lambda url: url.replace("https://demo-8017.euw.devtunnels.ms", URL)  # pas de vrai tunnel ici
            sent = subprocess.run(local(uploads[0]["command"]), shell=True, capture_output=True, text=True)
            check("envoi par curl", '"status":"stored"' in sent.stdout.replace(" ", ""), sent.stdout + sent.stderr)
            shared = call("chef", "get_validated_episode", episode_id=with_file["episode"]["id"])["episode"]["artifacts"][0]
            target = tmp / "poste_chef.xlsx"
            subprocess.run(["curl", "-sS", "-o", str(target), local(shared.get("download_url", URL))], check=False)
            check("un collègue du groupe télécharge le fichier", target.is_file() and target.read_bytes() == report.read_bytes())
            check("aucun chemin du serveur n'est exposé", shared["path"] is None and shared["stored_path"] is None)
            check("carol (RH) n'a pas accès au fichier", call("carol", "get_validated_episode", episode_id=with_file["episode"]["id"])["status"] == "not_found")
            check("lien sans ticket valide refusé", httpx.get(f"{URL}/files/dl/faux").status_code == 404)

            check("le chef revalide", call("chef", "store_validated_episode", question=Q, answer="v2", reasoning="x",
                                           confirmed_by_user=True, replace_episode_id=ep["id"])["status"] == "replaced")
            check("alice ne peut plus supprimer", call("alice", "delete_episode", episode_id=ep["id"], confirmed_by_user=True)["status"] == "forbidden")
            check("alice ne gère pas les utilisateurs", call("alice", "list_users")["status"] == "forbidden")
            created = call("chef", "save_user", user="bob@x.com", confirmed_by_user=True, groups=["finance"])
            check("le chef crée bob", created.get("status") == "created")
            tokens["bob"] = created.get("token", "")
            check("bob se connecte", call("bob", "whoami").get("groups") == ["FINANCE"])
            check("le chef supprime", call("chef", "delete_episode", episode_id=ep["id"], confirmed_by_user=True)["status"] == "deleted")
            print(f"\n{sum(checks)}/{len(checks)} vérifications réussies")
            if not all(checks):
                raise SystemExit(1)
        finally:
            server.terminate()
            server.wait(timeout=30)


if __name__ == "__main__":
    main()
