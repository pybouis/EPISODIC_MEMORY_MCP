# EPISODIC_MEMORY_MCP — mémoire épisodique pour assistant IA

Serveur MCP qui dote un assistant (Claude Code, Claude Desktop, tout client MCP) d'une
**mémoire épisodique** : une base de référence de paires question/réponse *validées par
l'utilisateur*, retrouvées par similarité vectorielle.

## Le protocole en 4 étapes

Le serveur transmet ce protocole à l'assistant via ses `instructions` MCP :

| # | Étape | Outil MCP |
|---|-------|-----------|
| 1 | **Comparer** la question du chat aux questions de la base (embedding + cosinus, plus mots-clés) | `find_similar_question` |
| 2 | **Proposer** dans le chat la question proche trouvée (si similarité ≥ seuil) | — (l'assistant demande) |
| 3 | Si validé, **intégrer** l'épisode stocké (réponse, raisonnement, fichiers) pour guider la réponse finale | `get_validated_episode` |
| 4 | **Proposer de valider** le nouvel épisode pour stockage, avec *qui* et *quand* | `store_validated_episode` |

## Contenu d'un épisode

Un épisode ne garde pas seulement le résultat, mais aussi le chemin qui y a mené :

- **`answer`** — la réponse validée. Si le résultat est un fichier, elle décrit son contenu.
- **`reasoning`** — la synthèse du cheminement de l'IA, rédigée par l'assistant à partir de
  tout l'historique de la conversation au moment du stockage. C'est une *fiche de reprise* en
  rubriques fixes : Contexte, Sources, Démarche, Décisions, Pièges, Reprise. À la
  réutilisation, l'assistant la traite comme l'historique d'une conversation qu'il reprend.
- **`artifacts`** — les fichiers produits (Excel, PDF, scripts…). Chacun est **copié** dans
  l'archive de la mémoire (`<data>\artifacts\<id épisode>\`) : il reste disponible même si
  l'original est déplacé ou supprimé, et peut resservir de modèle.
- **`keywords`** — les mots-clés du *sujet* de la question et leurs variantes (sigle,
  abréviation, synonyme, traduction), fournis par l'assistant au stockage. Ils servent à la
  recherche hybride décrite ci-dessous.

## Recherche hybride : vecteur + mots-clés

L'embedding seul rapproche mal les sigles et les questions courtes : « le PL SAFT 2022 » ne
fait que 0.61 avec « Donne moi le P&L de 2023 par division en euro », à peine plus qu'une
question hors sujet sur la même entité (0.50). La recherche combine donc deux signaux :

1. `vector_similarity` : le cosinus entre les embeddings des deux questions ;
2. un **bonus** (`EPISODIC_KEYWORD_BONUS`, 0.25) si un mot-clé de l'épisode figure, en terme
   entier, dans la question posée — à condition que le cosinus atteigne déjà un plancher
   (`EPISODIC_KEYWORD_FLOOR`, 0.45), pour qu'un mot-clé ne suffise pas à lui seul.

`similarity` (comparée au seuil) est la somme des deux, et `matched_keywords` liste les
mots-clés qui ont concordé. Avec l'exemple ci-dessus : 0.61 + 0.25 = 0.86, proposé.

**Mots-clés seuls.** Une question courte formulée avec un synonyme (« la trésorerie par
Unit » pour un épisode « analyser le cash par Unit… ») peut avoir un cosinus très faible
(0.15), sous le plancher. Quand un mot-clé concorde, le serveur calcule donc aussi
`keyword_coverage` : la part des mots porteurs de sens de la question (hors mots outils et
nombres) que l'on retrouve dans la question de l'épisode, ses mots-clés ou leurs synonymes.
Si elle atteint `EPISODIC_KEYWORD_COVERAGE` (0.75), `similarity` vaut au moins cette
couverture. Un seul mot commun ne suffit donc pas : « la trésorerie de SAFT » (couverture
0.5 face à l'épisode cash par Unit) n'est pas proposée par cette voie. Un score élevé avec
un `vector_similarity` faible signale un rapprochement par les mots, à confirmer avec soin.
Cette voie examine les 200 épisodes les plus proches vectoriellement dans le périmètre.

- **Normalisation** : la comparaison ignore la casse, les accents, la ponctuation et le
  pluriel en -s ; les sigles en « & » sont soudés (« P&L », « P & L » et « PL » sont le même terme).
- **Sujet, pas paramètres** : les mots-clés décrivent *de quoi* parle la question (P&L, cash,
  effets de change), pas l'entité, l'année ou la devise — sinon deux questions sans rapport
  sur la même entité seraient rapprochées. Ces paramètres vont dans `tags`, qui ne servent
  pas à la recherche.
- **Glossaire** (`<data>\glossary.json`) : groupes de termes équivalents, valables pour tous
  les épisodes — `{"synonyms": [["BFR", "besoin en fonds de roulement", "working capital"]]}`.
  Un épisode dont le mot-clé est « BFR » est alors retrouvé par « working capital ». Il se
  complète par l'outil `add_synonyms` (administrateurs), par `episodic-memory glossary add`
  ou à la main ; le fichier est relu à chaud.
- **Côté question** : l'assistant peut passer dans `keywords` de `find_similar_question`
  les autres formes des termes de la question (« PL » → « P&L », « compte de résultat »).
- **Enrichir un épisode existant** : outil `set_episode_keywords` ou
  `episodic-memory set-keywords <id> "P&L,PnL,compte de résultat"`. Aucun vecteur n'est
  recalculé : les mots-clés sont une simple métadonnée, incluse dans l'export JSON.
- **Ambiguïté** : un sigle peut désigner deux notions (« PL » : P&L ou Product Line). Le
  rapprochement par mot-clé reste une *proposition* : l'assistant montre `matched_keywords`
  et l'utilisateur confirme (étape 2). La détection de doublon au stockage, elle, reste
  purement vectorielle.

Maîtrise des tokens : la synthèse est bornée à **4000 caractères** (≈ 1000 tokens, cible
1500–2500). Au-delà, le serveur refuse le stockage (`reasoning_too_long`) et l'assistant doit
condenser. La recherche (étapes 1–2) ne renvoie ni réponse ni raisonnement : rien n'est
chargé dans le contexte avant votre validation.

À l'étape 4, l'assistant vous montre la synthèse et la liste des fichiers avant de stocker.

Garde-fous :

- `find_similar_question` ne renvoie **pas** la réponse stockée : elle n'est accessible
  qu'à l'étape 3, après accord de l'utilisateur.
- `store_validated_episode` et `delete_episode` refusent d'agir sans `confirmed_by_user=true`.
- Au stockage, une question quasi identique (similarité ≥ 0.95) est signalée : l'assistant
  demande s'il faut **remplacer** l'épisode (révision +1) ou en créer un nouveau.

Autres outils : `list_episodes`, `delete_episode`, `export_episodes`, `whoami`,
`set_episode_keywords`, `set_episode_usage`, `add_synonyms`.

Chaque épisode enregistre : `group`, `question`, `answer`, `reasoning`, `artifacts`, `validated_by`,
`validated_at`, `created_by`, `created_at`, `revision`, `use_count`, `last_used_at`, `tags`, `keywords`,
`usage` (tokens consommés, voir « Tokens consommés par épisode »).
En remplacement d'un épisode, le raisonnement, les mots-clés et les fichiers existants sont conservés si
l'assistant n'en fournit pas de nouveaux.

## Choix techniques

- **Base vectorielle** : ChromaDB persistant (distance cosinus).
- **Embedding** : local, sans clé d'API — `paraphrase-multilingual-MiniLM-L12-v2` via
  fastembed (ONNX, 384 dimensions, français/anglais et ~50 langues). Téléchargé une seule
  fois (~220 Mo), puis utilisé hors ligne.
- **SDK MCP** : `mcp` 2.x (`MCPServer`), transport stdio (poste local) ou HTTP streamable (serveur partagé).

## Emplacements

| Quoi | Où |
|------|----|
| Code | ce dossier (OneDrive) |
| Venv | `C:\dev_python\venv_episodic_memory_mcp` |
| Base vectorielle + modèle | `C:\dev_python\data_episodic_memory_mcp` |
| Archive des fichiers produits | `C:\dev_python\data_episodic_memory_mcp\artifacts` |
| Sauvegardes JSON | `exports\` (OneDrive) |

La base est volontairement **hors OneDrive** : la synchronisation de fichiers SQLite/HNSW
ouverts peut les corrompre. Pour sauvegarder, utiliser `export_episodes` (ou
`episodic-memory export`) : le JSON arrive dans `exports\`, synchronisé par OneDrive.

L'export JSON contient les questions, réponses, raisonnements et les *références* des
fichiers, **pas les fichiers archivés eux-mêmes** : pour les sauvegarder, copier le dossier
`artifacts` (les fichiers de plus de 50 Mo ne sont pas archivés, seul leur chemin d'origine
est mémorisé).

## Brancher sur Claude Code

```powershell
claude mcp add --scope user episodic-memory --env EPISODIC_USER=prenom.nom@exemple.com -- C:\dev_python\venv_episodic_memory_mcp\Scripts\python.exe -m episodic_memory
```

`EPISODIC_USER` est l'identité inscrite dans `validated_by` (défaut : login Windows).
Si `claude` n'est pas dans le PATH, l'exécutable est `%USERPROFILE%\.local\bin\claude.exe`.
Vérifier : `claude mcp get episodic-memory` ; retirer : `claude mcp remove episodic-memory -s user`.

### Rappel automatique (hook `UserPromptSubmit`)

Un serveur MCP actif ne suffit pas : ses instructions sont indicatives et l'assistant peut
les ignorer (constaté : aucune recherche ni proposition de stockage). Le hook
`episodic_memory.hook` est exécuté par Claude Code à chaque message envoyé et injecte un
court rappel du protocole dans le contexte. À déclarer dans `~/.claude/settings.json` :

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "C:\\dev_python\\venv_episodic_memory_mcp\\Scripts\\python.exe",
            "args": ["-m", "episodic_memory.hook"],
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

- Rien n'est injecté pour un message de moins de 4 mots (« oui », « ok merci ») ni pour
  une commande `/…`.
- Coût : ~1000 caractères (≈ 250 tokens) par message concerné, ~160 ms d'exécution. Le
  hook ne charge ni la base ni le modèle d'embedding.
- Le hook garantit que le rappel est présent ; c'est toujours l'assistant qui appelle les
  outils. Un modèle plus capable suit le protocole plus fidèlement.
- Désactiver ponctuellement : variable d'environnement `EPISODIC_HOOK_DISABLED=1` ;
  définitivement : retirer le bloc `hooks` de `settings.json`.

### Tokens consommés par épisode

Chaque épisode porte deux mesures :

| Champ | Quoi | Comment |
|-------|------|---------|
| `size_tokens` | Taille de l'épisode : question, réponse, raisonnement (≈ coût d'une réutilisation) | Estimée à la lecture (`EPISODIC_CHARS_PER_TOKEN`, 4 caractères/token) : vaut aussi pour les anciens épisodes |
| `usage` | Tokens réellement consommés par l'IA pour produire la réponse : `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `total_tokens`, `fresh_tokens` (hors relectures du cache), `api_calls`, `models`, fenêtre mesurée | Lue dans le journal de la conversation Claude Code (`~/.claude/projects/<projet>/<session>.jsonl`) |

`list_episodes` et `find_similar_question` renvoient pour chaque épisode `tokens` =
`{size, consumed, output}` ; `list_episodes` ajoute `token_totals` sur tout le périmètre.
En CLI : `episodic-memory tokens [--group G]`.

**Fenêtre mesurée** : du message de l'utilisateur qui précède l'appel à `find_similar_question`
le plus proche de la question stockée, jusqu'à l'appel à `store_validated_episode` (sous-agents
compris, chaque réponse de l'API comptée une fois). Jamais avant le stockage d'une autre
question dans la même conversation. Sans recherche préalable : depuis le début de la
conversation (`window: "session"`) ou depuis le stockage précédent (`since_previous_store`).
Les relectures du cache (`cache_read_input_tokens`, le contexte relu à chaque appel) font
souvent plus de 90 % du total : regarder aussi `output_tokens` et `fresh_tokens`.

**Qui mesure** :

- *stdio (poste local)* : le serveur lit lui-même le journal au moment du stockage
  (`EPISODIC_TRACK_USAGE`, actif par défaut en stdio ; `EPISODIC_CLAUDE_PROJECTS_DIR`).
- *http (serveur partagé)* : le serveur ne voit pas le poste de l'utilisateur. Un hook
  `PostToolUse` mesure le journal côté client après chaque stockage réussi et demande à
  l'assistant d'appeler `set_episode_usage` avec les chiffres. À ajouter dans
  `~/.claude/settings.json`, à côté du hook `UserPromptSubmit` :

```json
"PostToolUse": [
  {
    "matcher": "mcp__episodic-memory__store_validated_episode",
    "hooks": [
      {
        "type": "command",
        "command": "C:\\dev_python\\venv_episodic_memory_mcp\\Scripts\\python.exe",
        "args": ["-m", "episodic_memory.hook"],
        "timeout": 15
      }
    ]
  }
]
```

- *autre client MCP* : `store_validated_episode` accepte un paramètre `usage` facultatif, à ne
  remplir que si le client connaît réellement ses chiffres.

**Épisodes déjà stockés** : `episodic-memory backfill-usage --dry-run` retrouve dans les
journaux locaux les appels de stockage passés (même texte de question) et affiche la mesure ;
sans `--dry-run`, il l'enregistre. Par défaut seules les fenêtres fiables (`question`) sont
retenues ; `--any-window` accepte les autres, `--force` remesure les épisodes déjà mesurés.

Claude Desktop (`claude_desktop_config.json`) :

```json
{
  "mcpServers": {
    "episodic-memory": {
      "command": "C:\\dev_python\\venv_episodic_memory_mcp\\Scripts\\python.exe",
      "args": ["-m", "episodic_memory"],
      "env": { "EPISODIC_USER": "prenom.nom@exemple.com" }
    }
  }
}
```

## Serveur partagé et sécurité

Deux modes, choisis par `EPISODIC_TRANSPORT` :

- **`stdio`** (défaut) — poste local, un seul utilisateur. Il possède le processus et les
  fichiers de la base : il est administrateur de sa propre mémoire, sans authentification.
- **`http`** — serveur partagé (HTTP streamable sur `/mcp`). Le serveur est le seul à accéder
  à la base ; chaque appel est authentifié et limité aux groupes de l'appelant.

Chaque épisode appartient à un **groupe**. Les rôles :

| Action | Admin | Admin de groupe | Utilisateur |
|--------|-------|-----------------|-------------|
| Chercher / lire | tout | ses groupes | ses groupes |
| Stocker | tout groupe | ses groupes | ses groupes |
| Remplacer | tout | son groupe | les épisodes qu'il a créés |
| Supprimer | tout | son groupe | ceux qu'il a créés **et** validés |
| Exporter | tout | son groupe | non |
| Gérer les utilisateurs (variante `local`) | tout | simples membres de son groupe | non |

- L'identité inscrite dans `validated_by` est celle de l'appelant authentifié ; elle ne peut
  plus être passée en paramètre.
- Le filtre de groupe est appliqué **dans** la requête vectorielle : recherche, compteur et
  détection de doublon ne voient jamais un épisode d'un autre groupe. Un épisode hors
  périmètre est « introuvable ».
- Si un administrateur de groupe revalide (remplace) un épisode, son auteur perd le droit
  de le supprimer : il ne l'a plus « créé et validé ».
- Un utilisateur membre de plusieurs groupes doit préciser `group` au stockage (le serveur
  répond `group_required` avec la liste).
- Sur un serveur partagé, les fichiers produits sont **stockés sur le serveur** et
  accessibles à tout le groupe de l'épisode (voir « Fichiers produits partagés »).
- Les stockages, remplacements, suppressions, exports et changements d'utilisateurs sont
  tracés dans `<data>\logs\episodic_memory_mcp.log` (jamais les jetons).

### Fichiers produits partagés

Le serveur ne voit pas les fichiers du poste de l'auteur, et un assistant ne peut pas faire
passer un Excel ou un PDF dans un appel d'outil. Les fichiers transitent donc par HTTP,
avec des liens temporaires délivrés à un appelant dont les droits viennent d'être vérifiés :

1. `store_validated_episode` enregistre l'épisode et renvoie `uploads` : pour chaque fichier,
   une commande `curl -sS -T "<fichier local>" "<lien d'envoi>"` que l'assistant exécute sur
   le poste de l'auteur. Le lien est à usage unique.
2. Le serveur range le fichier dans `<data>\artifacts\<id épisode>\` (maximum
   `EPISODIC_ARTIFACT_MAX_MB`, 50 Mo par défaut).
3. `get_validated_episode`, pour tout membre du groupe, renvoie un `download_url` par
   fichier : l'assistant le télécharge avec `curl -o`, ou donne le lien à l'utilisateur.

Les liens expirent après `EPISODIC_FILE_LINK_MINUTES` (30 par défaut) et sont perdus au
redémarrage du serveur ; il suffit de rappeler l'outil pour en obtenir de nouveaux. Aucun
chemin du serveur n'est jamais renvoyé. Supprimer un épisode supprime ses fichiers.

Les fichiers ne sont pas dans l'export JSON : pour reprendre ceux d'une base locale,
remplacer l'épisode depuis le poste qui les possède (les liens d'envoi sont renvoyés pour
tout fichier que le serveur n'a pas encore reçu). Sauvegarde : copier le dossier `artifacts`.

La variante se choisit dans le `.env` du serveur avec `EPISODIC_AUTH_PROVIDER`.

### Variante `local` — utilisateurs dans la base de la mémoire

Utilisateurs, groupes et rôles sont stockés dans la même base ChromaDB (collection
`security_users`). Chaque utilisateur a un jeton personnel `emk_…` ; seul son hash SHA-256
est stocké, le jeton n'est affiché qu'une fois.

```powershell
$em = "C:\episodic_memory_mcp\episodic-memory-mcp\episodic-memory-mcp.exe"
& $em user add pierre@exemple.com --admin                      # administrateur de la base
& $em user add chef@exemple.com --admin-of FINANCE             # administrateur du groupe FINANCE
& $em user add alice@exemple.com --groups FINANCE,RH           # simple membre
& $em user set alice@exemple.com --groups FINANCE --disable    # modifier / désactiver
& $em user token alice@exemple.com                             # nouveau jeton (révoque l'ancien)
& $em user list
```

Arrêter le serveur avant d'écrire en ligne de commande (un seul processus écrivain). Serveur
en marche, les administrateurs passent par les outils MCP `list_users`, `save_user` et
`reset_user_token` ; `whoami` indique à chacun son rôle et ses groupes.

### Variante `pa` — sécurité de Planning Analytics

Le client envoie le jeton OAuth Planning Analytics de l'utilisateur ; le serveur le présente
à l'API REST de PA (`<EPISODIC_PA_URL>/ActiveUser`, avec ses groupes) et en déduit le rôle.
Aucun utilisateur ni mot de passe n'est géré côté mémoire ; le résultat est gardé en cache
5 minutes par jeton (`EPISODIC_PA_CACHE_SECONDS`). Le serveur de la mémoire n'a pas besoin
d'être sur la machine de PA : il lui suffit de joindre son URL.

| Groupe Planning Analytics | Rôle dans la mémoire |
|---------------------------|----------------------|
| `EPISODIC_ADMIN` (liste `EPISODIC_PA_ADMIN_GROUPS`) | administrateur de toute la base |
| `EPISODIC_<GROUPE>_ADMIN` | administrateur du groupe `<GROUPE>` |
| `EPISODIC_<GROUPE>` | membre du groupe `<GROUPE>` |

Avec `EPISODIC_PA_GROUP_PREFIX=` (vide), tous les groupes PA existants (hors groupes système
`}…`) deviennent des groupes de la mémoire ; ajouter alors `EPISODIC_PA_ADMIN_GROUPS=ADMIN`
pour que les administrateurs PA administrent la mémoire.

Limites : le serveur dépend de la disponibilité de PA (réponse 503 sinon), et le client MCP
doit savoir obtenir le jeton OAuth de PA (c'est le cas d'une connexion OAuth watsonx
Orchestrate ; Claude Code doit recevoir le jeton dans un en-tête `Authorization`).

## Livraison sur un serveur Windows (.exe + Dev Tunnel)

1. **Construire** sur le poste de développement (sortie hors OneDrive) :

   ```powershell
   .\scripts\build_exe.ps1 -WithModel
   ```

   Livrable : `C:\dev_python\build_episodic_memory_mcp\dist\episodic-memory-mcp\` (l'exe,
   `_internal\`, `.env.example`, `install_mcp_task.ps1` et, avec `-WithModel`, le modèle
   d'embedding dans `models\` pour un serveur sans accès à Hugging Face).
2. **Copier** ce dossier sur le serveur, par exemple dans `C:\episodic_memory_mcp\`.
3. **Configurer** : copier `.env.example` en `.env` à côté de l'exe et renseigner le port,
   `MCP_PUBLIC_BASE_URL` (URL du tunnel), `EPISODIC_DATA_DIR` et `EPISODIC_AUTH_PROVIDER`.
4. **Reprendre la base** existante : `episodic-memory-mcp.exe import episodes_….json`
   (un export contient le groupe de chaque épisode).
5. **Créer le premier administrateur** (variante `local`) :
   `episodic-memory-mcp.exe user add vous@exemple.com --admin` et noter le jeton affiché.
6. **Démarrer au boot**, sans session ouverte, dans une console administrateur :
   `.\install_mcp_task.ps1` (tâche planifiée sous SYSTEM, relancée en cas d'échec, test de
   `/health`). Désinstaller : `.\install_mcp_task.ps1 -Uninstall`.
7. **Ouvrir le port dans le tunnel** qui tourne déjà sur le serveur, puis relancer son hôte :

   ```powershell
   devtunnel port create <id-du-tunnel> -p 8010
   devtunnel access create <id-du-tunnel> -p 8010 --anonymous
   devtunnel host <id-du-tunnel>
   ```

   L'accès anonyme est nécessaire : un client MCP ne sait pas s'authentifier auprès de
   Dev Tunnels, c'est le jeton de la mémoire qui protège le serveur. L'hôte public
   (`https://<id>-8010.<région>.devtunnels.ms`) doit être celui de `MCP_PUBLIC_BASE_URL`,
   sinon le serveur répond 421.
8. **Brancher un client** (Claude Code) :

   ```powershell
   claude mcp add --scope user --transport http episodic-memory https://<id>-8010.<région>.devtunnels.ms/mcp --header "Authorization: Bearer emk_…"
   ```

Sans argument, l'exe lance le serveur ; avec des arguments, c'est la ligne de commande
(`episodic-memory-mcp.exe --help`). Vérification complète d'un livrable sur une base
temporaire : `python tests\e2e_http_security.py <chemin de l'exe>`.

## Configuration (`.env` à côté de l'exe ou du projet, puis variables d'environnement)

| Variable | Défaut | Rôle |
|----------|--------|------|
| `EPISODIC_USER` | login Windows | identité du valideur en mode `stdio` |
| `EPISODIC_TRANSPORT` | `stdio` | `stdio` (local) ou `http` (serveur partagé) |
| `MCP_HOST` / `MCP_PORT` | `127.0.0.1` / `8010` | écoute du serveur HTTP |
| `MCP_PUBLIC_BASE_URL` | — | URL publique du tunnel (hôte accepté dans l'en-tête `Host`) |
| `EPISODIC_AUTH_PROVIDER` | `none` | `local` ou `pa` ; obligatoire en `http` |
| `MCP_ALLOW_NO_AUTH` | `false` | `http` sans authentification (test local : tout appelant est admin) |
| `EPISODIC_DEFAULT_GROUP` | `ADMIN` | groupe d'un épisode stocké sans groupe (mode local, admin sans groupe) |
| `EPISODIC_PA_URL` | — | racine REST de la base TM1 (`…/api/<tenant>/v0/tm1/<base>`) |
| `EPISODIC_PA_GROUP_PREFIX` | `EPISODIC_` | préfixe des groupes PA reconnus (vide = tous) |
| `EPISODIC_PA_GROUP_ADMIN_SUFFIX` | `_ADMIN` | suffixe des groupes PA d'administrateurs de groupe |
| `EPISODIC_PA_ADMIN_GROUPS` | `EPISODIC_ADMIN` | groupes PA administrateurs de toute la mémoire |
| `EPISODIC_PA_CACHE_SECONDS` | `300` | durée de cache des groupes d'un jeton PA |
| `EPISODIC_FILE_LINK_MINUTES` | `30` | validité des liens d'envoi et de téléchargement des fichiers |
| `EPISODIC_LOG_DIR` | `<data>\logs` | journal du serveur HTTP |
| `EPISODIC_ENV_FILE` | `.env` à côté de l'exe | autre fichier de configuration |
| `EPISODIC_THRESHOLD` | `0.75` | similarité minimale pour proposer une question |
| `EPISODIC_KEYWORD_BONUS` | `0.25` | bonus de similarité quand un mot-clé de l'épisode figure dans la question (`0` = recherche purement vectorielle) |
| `EPISODIC_KEYWORD_FLOOR` | `0.45` | similarité vectorielle minimale pour accorder ce bonus |
| `EPISODIC_KEYWORD_COVERAGE` | `0.75` | part minimale des mots de la question retrouvés dans l'épisode pour qu'un mot-clé concordant suffise sans le vecteur (`> 1` = désactivé) |
| `EPISODIC_GLOSSARY` | `<data>\glossary.json` | glossaire de synonymes partagé |
| `EPISODIC_DUPLICATE_THRESHOLD` | `0.95` | similarité à partir de laquelle on parle de doublon |
| `EPISODIC_REASONING_MAX_CHARS` | `4000` | taille maximale de la synthèse du raisonnement |
| `EPISODIC_ARTIFACT_MAX_MB` | `50` | taille maximale d'un fichier copié dans l'archive |
| `EPISODIC_ARTIFACTS_DIR` | `<data>\artifacts` | archive des fichiers produits |
| `EPISODIC_DATA_DIR` | `C:\dev_python\data_episodic_memory_mcp` | racine base + modèle + archive |
| `EPISODIC_DB_DIR` | `<data>\chroma` | base ChromaDB |
| `EPISODIC_MODEL_CACHE` | `<data>\models` | cache du modèle d'embedding |
| `EPISODIC_EXPORT_DIR` | `<projet>\exports` | sauvegardes JSON |
| `EPISODIC_EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | modèle fastembed |

Repères de similarité mesurés avec le modèle par défaut : paraphrase en français ≈ 0.95,
même question en anglais ≈ 0.79, même domaine mais autre question ≈ 0.47, sans rapport ≈ 0.10.

Changer de modèle d'embedding sur une base existante : exporter, supprimer
`<data>\chroma`, puis réimporter (les vecteurs sont recalculés).

## Ligne de commande

```powershell
$em = "C:\dev_python\venv_episodic_memory_mcp\Scripts\episodic-memory.exe"
& $em list
& $em search "Comment créer un cube TM1 ?"
& $em search "le PL SAFT 2022" --keywords "P&L,compte de résultat"
& $em set-keywords ep_xxxxxxxxxxxx "P&L,PnL,compte de résultat"   # liste complète des mots-clés de l'épisode
& $em glossary list
& $em glossary add "BFR,besoin en fonds de roulement,working capital"
& $em show ep_xxxxxxxxxxxx
& $em export
& $em import exports\episodes_20261109_214500.json
& $em delete ep_xxxxxxxxxxxx
& $em assign-group ADMIN          # rattache à un groupe les épisodes qui n'en ont pas
```

Ne pas lancer d'écriture en ligne de commande pendant qu'une session de l'assistant
utilise la base : ChromaDB persistant n'est pas prévu pour plusieurs processus écrivains.

## Développement

```powershell
$py = "C:\dev_python\venv_episodic_memory_mcp\Scripts\python.exe"
& $py -m pytest                 # tests unitaires (embedding factice, sans réseau)
& $py tests\e2e_mcp_demo.py     # démo de bout en bout : vrai serveur stdio + vrai modèle, base temporaire
& $py tests\e2e_http_security.py  # serveur HTTP partagé : jetons, rôles et groupes, base temporaire
```

Recréer le venv :

```powershell
python -m venv C:\dev_python\venv_episodic_memory_mcp
C:\dev_python\venv_episodic_memory_mcp\Scripts\python.exe -m pip install -e ".[dev]"
```

Premier démarrage du serveur : ~13 s (chargement de ChromaDB et du modèle, fait en
arrière-plan) ; ensuite une recherche prend ~20 ms.
