param(
    [string]$RunName = "banana_glasses_v7_sil_b2048_e4_a10",
    [int]$NumActors = 10,
    [int]$MaxUpdates = 0,
    [int]$FragmentSteps = 0,
    [int]$GlobalRolloutSteps = 0,
    [string]$Config = "",
    [string]$Python = "python",
    [switch]$UseRootConfig,
    [switch]$DryRun,
    [switch]$LocalMode
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return (Resolve-Path $Path).Path
    }
    return (Resolve-Path (Join-Path $RepoRoot $Path)).Path
}

function Resolve-RunConfig {
    if ($Config) {
        return Resolve-RepoPath $Config
    }

    $runConfig = Join-Path $RepoRoot (Join-Path "models" (Join-Path $RunName "config.yaml"))
    if (-not $UseRootConfig -and (Test-Path -LiteralPath $runConfig)) {
        return (Resolve-Path $runConfig).Path
    }

    return (Resolve-Path (Join-Path $RepoRoot "config.yaml")).Path
}

$ConfigPath = Resolve-RunConfig
$oldConfigPath = $env:SNN_CONFIG_PATH
$exitCode = 0

try {
    Push-Location $RepoRoot
    $env:SNN_CONFIG_PATH = $ConfigPath

    $trainArgs = @(
        "-m", "distributed.ray_train",
        "--config", $ConfigPath,
        "--run-name", $RunName,
        "--num-actors", "$NumActors"
    )
    if ($MaxUpdates -gt 0) {
        $trainArgs += @("--max-updates", "$MaxUpdates")
    }
    if ($FragmentSteps -gt 0) {
        $trainArgs += @("--fragment-steps", "$FragmentSteps")
    }
    if ($GlobalRolloutSteps -gt 0) {
        $trainArgs += @("--global-rollout-steps", "$GlobalRolloutSteps")
    }
    if ($LocalMode) {
        $trainArgs += "--local-mode"
    }

    Write-Host "Launching Ray SIL run: $RunName"
    Write-Host "Repo: $RepoRoot"
    Write-Host "Config: $ConfigPath"
    Write-Host "Command: $Python $($trainArgs -join ' ')"
    if (-not $DryRun) {
        & $Python @trainArgs
        $exitCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
    if ($null -eq $oldConfigPath) {
        Remove-Item Env:SNN_CONFIG_PATH -ErrorAction SilentlyContinue
    }
    else {
        $env:SNN_CONFIG_PATH = $oldConfigPath
    }
}

exit $exitCode
