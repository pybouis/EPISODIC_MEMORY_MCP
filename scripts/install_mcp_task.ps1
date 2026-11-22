<#
.SYNOPSIS
    Lance episodic-memory-mcp.exe au démarrage de Windows, sans session ouverte (tâche planifiée).

.DESCRIPTION
    Crée une tâche planifiée qui démarre le serveur MCP au boot sous le compte SYSTEM, avec le
    dossier de l'exe comme répertoire de travail (le .env posé à côté de l'exe est ainsi trouvé),
    sans limite de durée d'exécution, et relancée toutes les minutes en cas d'échec.
    À exécuter dans une console PowerShell « Exécuter en tant qu'administrateur ».

.EXAMPLE
    .\install_mcp_task.ps1                        # exe dans le dossier du script
.EXAMPLE
    .\install_mcp_task.ps1 -ExeDir D:\episodic-memory\episodic-memory-mcp
.EXAMPLE
    .\install_mcp_task.ps1 -Uninstall
#>
param(
    [string]$ExeDir = $PSScriptRoot,
    [string]$TaskName = "episodic-memory-mcp",
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "Lancez ce script dans une console PowerShell ouverte en tant qu'administrateur."
}

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Get-Process episodic-memory-mcp -ErrorAction SilentlyContinue | Stop-Process -Force
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Tâche '$TaskName' supprimée et serveur arrêté."
    } else {
        Write-Host "Aucune tâche '$TaskName'."
    }
    return
}

# ------------------------------------------------------------------ vérifications
$ExeDir = (Resolve-Path $ExeDir).Path
$exe = Join-Path $ExeDir "episodic-memory-mcp.exe"
if (-not (Test-Path $exe)) { throw "Exe introuvable : $exe (précisez -ExeDir)." }

$envFile = [Environment]::GetEnvironmentVariable("EPISODIC_ENV_FILE", "Machine")
if (-not $envFile) { $envFile = Join-Path $ExeDir ".env" }
if (-not (Test-Path $envFile)) {
    throw "Fichier .env introuvable : $envFile (posez-le à côté de l'exe ou définissez EPISODIC_ENV_FILE au niveau Machine)."
}
if ([Environment]::GetEnvironmentVariable("EPISODIC_ENV_FILE", "User")) {
    Write-Warning "EPISODIC_ENV_FILE est défini au niveau Utilisateur : le compte SYSTEM ne le voit pas, définissez-le au niveau Machine."
}

# Port et hôte lus dans le .env, pour le test /health final.
$values = @{}
Get-Content $envFile | Where-Object { $_ -match '^\s*(MCP_PORT|PORT|MCP_HOST)\s*=\s*(.*?)\s*$' } |
    ForEach-Object { $values[$Matches[1]] = $Matches[2].Trim('"', "'") }
$port = if ($values.MCP_PORT) { $values.MCP_PORT } elseif ($values.PORT) { $values.PORT } else { "8010" }
$hostName = if ($values.MCP_HOST -and $values.MCP_HOST -notin "0.0.0.0", "::") { $values.MCP_HOST } else { "127.0.0.1" }

# ------------------------------------------------------------------ tâche planifiée
$action = New-ScheduledTaskAction -Execute $exe -WorkingDirectory $ExeDir
$trigger = New-ScheduledTaskTrigger -AtStartup
# ExecutionTimeLimit à zéro : sinon Windows arrête la tâche au bout de 72 h par défaut.
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Serveur MCP episodic-memory (démarrage automatique)" -Force | Out-Null
Write-Host "Tâche '$TaskName' créée : $exe (répertoire de travail $ExeDir, compte SYSTEM)."

# ------------------------------------------------------------------ démarrage et test
Get-Process episodic-memory-mcp -ErrorAction SilentlyContinue | Stop-Process -Force  # instance lancée à la main
Start-ScheduledTask -TaskName $TaskName
$url = "http://${hostName}:$port/health"
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod $url -TimeoutSec 2
        Write-Host "Serveur démarré : $url -> $($health.status) (version $($health.version))"
        return
    } catch {}
}
$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Warning "Pas de réponse sur $url après 30 s (dernier résultat de la tâche : $($info.LastTaskResult)). Consultez le dossier log du projet."
