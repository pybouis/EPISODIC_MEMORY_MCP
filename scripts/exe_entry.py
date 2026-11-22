"""Point d'entrée de l'exe : sans argument, lance le serveur ; sinon, la ligne de commande.

    episodic-memory-mcp.exe                      -> serveur MCP (transport selon le .env)
    episodic-memory-mcp.exe user add …           -> administration (voir `--help`)
"""

import sys

if len(sys.argv) > 1:
    from episodic_memory.cli import main
else:
    from episodic_memory.server import main

main()
