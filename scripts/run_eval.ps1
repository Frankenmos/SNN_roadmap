# Simple eval launcher. Run it, type the model (run) name when asked, and it
# loads that run's checkpoint from models/<run_name>/ and plays with the SC2
# window visible so you can watch behaviour.
#
#   .\scripts\run_eval.ps1                 # normal checkpoint, deterministic
#   .\scripts\run_eval.ps1 -Stochastic     # sample actions instead of argmax
#   .\scripts\run_eval.ps1 -Best           # use best_checkpoint.pth
param(
    [switch]$Stochastic,   # default is deterministic (argmax)
    [switch]$Best,         # default is the normal checkpoint.pth
    [int]$Episodes = 5,
    [string]$RunName = "",
    [string]$Python = "python.exe"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $RunName) {
    $RunName = Read-Host "Model (run) name"
}
$RunName = $RunName.Trim()
if (-not $RunName) {
    throw "No model name given."
}

$RunDir = Join-Path $RepoRoot (Join-Path "models" $RunName)
if (-not (Test-Path -LiteralPath $RunDir)) {
    throw "Model folder not found: $RunDir"
}

$evalArgs = @("eval.py", "--run_name", $RunName, "--episodes", "$Episodes")
if ($Best) { $evalArgs += "--best" }
if ($Stochastic) { $evalArgs += "--nodeterministic" } else { $evalArgs += "--deterministic" }

if ($Best) { $ckptLabel = "best_checkpoint.pth" } else { $ckptLabel = "checkpoint.pth" }
if ($Stochastic) { $modeLabel = "stochastic" } else { $modeLabel = "deterministic" }
Write-Host "Run:        $RunName"
Write-Host "Checkpoint: $ckptLabel"
Write-Host "Mode:       $modeLabel ($Episodes episodes)"
Write-Host "Command:    $Python $($evalArgs -join ' ')"

try {
    Push-Location $RepoRoot
    & $Python @evalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "eval.py exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
