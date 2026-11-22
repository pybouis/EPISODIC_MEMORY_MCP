<#
.SYNOPSIS
    Construit episodic-memory-mcp.exe (PyInstaller, dossier « onedir ») prêt à copier sur le serveur.

.DESCRIPTION
    Le dossier de sortie est HORS OneDrive (la synchronisation verrouille les fichiers pendant
    la construction). Résultat : <OutDir>\dist\episodic-memory-mcp\ contenant l'exe, _internal\,
    .env.example et install_mcp_task.ps1. Avec -WithModel, le modèle d'embedding déjà téléchargé
    sur ce poste est copié dans le livrable (dossier models\) : utile si le serveur n'a pas
    accès à Hugging Face.

.EXAMPLE
    .\scripts\build_exe.ps1
.EXAMPLE
    .\scripts\build_exe.ps1 -WithModel
#>
param(
    [string]$Python = "C:\dev_python\venv_episodic_memory_mcp\Scripts\python.exe",
    [string]$OutDir = "C:\dev_python\build_episodic_memory_mcp",
    [string]$ModelDir = "C:\dev_python\data_episodic_memory_mcp\models",
    [switch]$WithModel
)

$ErrorActionPreference = "Stop"
$project = Split-Path $PSScriptRoot -Parent

# PyInstaller écrit son journal sur stderr : sous Windows PowerShell 5.1, « Stop » en ferait
# une erreur fatale dès la première ligne. Le code de sortie est testé explicitement.
$ErrorActionPreference = "Continue"
& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) { & $Python -m pip install pyinstaller }

Push-Location $project
try {
    & $Python -m PyInstaller --noconfirm --clean `
        --distpath (Join-Path $OutDir "dist") --workpath (Join-Path $OutDir "build") `
        episodic-memory-mcp.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller a échoué (code $LASTEXITCODE)." }
} finally {
    Pop-Location
}
$ErrorActionPreference = "Stop"

$dist = Join-Path $OutDir "dist\episodic-memory-mcp"
Copy-Item (Join-Path $project ".env.example") $dist -Force
Copy-Item (Join-Path $project "scripts\install_mcp_task.ps1") $dist -Force
if ($WithModel) {
    if (-not (Test-Path $ModelDir)) { throw "Modèle introuvable : $ModelDir" }
    Copy-Item $ModelDir (Join-Path $dist "models") -Recurse -Force
}
Write-Host "Livrable : $dist"
