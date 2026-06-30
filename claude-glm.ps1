<#
.SYNOPSIS
Launch Claude Code through a ZAI/GLM-compatible Anthropic endpoint.

.DESCRIPTION
Sets GLM/ZAI model-routing environment variables only for the duration of the
Claude Code process, then restores the terminal's previous environment.

Recommended setup:
  .\tools\claude-glm.ps1

The script uses https://api.z.ai/api/anthropic by default and prompts for the
ZAI API key when it is not supplied. To avoid the prompt for one terminal only:
  $env:ZAI_API_KEY = "your-zai-api-key"
  .\tools\claude-glm.ps1

Additional Claude arguments can be forwarded with -ClaudeArgs:
  .\tools\claude-glm.ps1 -ClaudeArgs '--continue'
  .\tools\claude-glm.ps1 -Model opus -ClaudeArgs '--dangerously-skip-permissions'
#>

[CmdletBinding()]
param(
    [string]$BaseUrl = $(if ([string]::IsNullOrWhiteSpace($env:ZAI_ANTHROPIC_BASE_URL)) { "https://api.z.ai/api/anthropic" } else { $env:ZAI_ANTHROPIC_BASE_URL }),

    [string]$ApiKey = $env:ZAI_API_KEY,

    [ValidateSet("AuthToken", "ApiKey", "Both")]
    [string]$CredentialMode = "AuthToken",

    [string]$Model = "sonnet",

    [ValidateSet("default", "low", "medium", "high", "xhigh", "max", "ultracode")]
    [string]$Effort = "max",

    [string]$HaikuModel = "glm-4.7",

    [string]$SonnetModel = "glm-5.2[1m]",

    [string]$OpusModel = "glm-5.2[1m]",

    [switch]$NoCredential,

    [switch]$DryRun,

    [string[]]$ClaudeArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$parsedBaseUrl = $null
if (-not [Uri]::TryCreate($BaseUrl, [UriKind]::Absolute, [ref]$parsedBaseUrl) -or
    ($parsedBaseUrl.Scheme -ne "http" -and $parsedBaseUrl.Scheme -ne "https")) {
    throw "BaseUrl must be an absolute http(s) URL. Received: $BaseUrl"
}

$claudeCommand = Get-Command claude -ErrorAction SilentlyContinue
if ($null -eq $claudeCommand) {
    throw "Could not find 'claude' on PATH. Install or update Claude Code, then open a new terminal."
}

$keysToRestore = @(
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_DESCRIPTION",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION",
    "ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES",
    "ANTHROPIC_DEFAULT_OPUS_MODEL_SUPPORTED_CAPABILITIES",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_CODE_ALWAYS_ENABLE_EFFORT",
    "CLAUDE_CODE_EFFORT_LEVEL"
)

$originalEnv = @{}
foreach ($key in $keysToRestore) {
    $originalEnv[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
}

function Set-ProcessEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [AllowNull()]
        [string]$Value
    )

    [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
}

function Restore-ProcessEnv {
    foreach ($key in $keysToRestore) {
        Set-ProcessEnv -Name $key -Value $originalEnv[$key]
    }
}

$apiKeyBstr = [IntPtr]::Zero
$apiKeyWasProvided = -not [string]::IsNullOrWhiteSpace($ApiKey)

try {
    if (-not $NoCredential -and [string]::IsNullOrWhiteSpace($ApiKey)) {
        if ($DryRun) {
            $ApiKey = "dry-run-placeholder"
        }
        else {
            $secureApiKey = Read-Host -Prompt "ZAI API key" -AsSecureString
            $apiKeyBstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureApiKey)
            $ApiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($apiKeyBstr)

            if ([string]::IsNullOrWhiteSpace($ApiKey)) {
                throw "ZAI API key is required. Pass -ApiKey, set ZAI_API_KEY for this terminal, or use -NoCredential."
            }
        }
    }

    Set-ProcessEnv -Name "ANTHROPIC_BASE_URL" -Value $BaseUrl

    Set-ProcessEnv -Name "ANTHROPIC_AUTH_TOKEN" -Value $null
    Set-ProcessEnv -Name "ANTHROPIC_API_KEY" -Value $null

    if (-not $NoCredential) {
        if ($CredentialMode -eq "AuthToken" -or $CredentialMode -eq "Both") {
            Set-ProcessEnv -Name "ANTHROPIC_AUTH_TOKEN" -Value $ApiKey
        }
        if ($CredentialMode -eq "ApiKey" -or $CredentialMode -eq "Both") {
            Set-ProcessEnv -Name "ANTHROPIC_API_KEY" -Value $ApiKey
        }
    }

    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_HAIKU_MODEL" -Value $HaikuModel
    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_SONNET_MODEL" -Value $SonnetModel
    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_OPUS_MODEL" -Value $OpusModel
    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME" -Value "GLM 5.2 1M"
    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME" -Value "GLM 5.2 1M"
    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_SONNET_MODEL_DESCRIPTION" -Value "GLM-5.2 routed through ZAI"
    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION" -Value "GLM-5.2 routed through ZAI"

    $glmCapabilities = "effort,xhigh_effort,max_effort,thinking,adaptive_thinking,interleaved_thinking"
    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES" -Value $glmCapabilities
    Set-ProcessEnv -Name "ANTHROPIC_DEFAULT_OPUS_MODEL_SUPPORTED_CAPABILITIES" -Value $glmCapabilities

    Set-ProcessEnv -Name "CLAUDE_CODE_AUTO_COMPACT_WINDOW" -Value "1000000"
    Set-ProcessEnv -Name "CLAUDE_CODE_ALWAYS_ENABLE_EFFORT" -Value "1"

    if ($Effort -eq "default") {
        Set-ProcessEnv -Name "CLAUDE_CODE_EFFORT_LEVEL" -Value $null
    }
    else {
        Set-ProcessEnv -Name "CLAUDE_CODE_EFFORT_LEVEL" -Value $Effort
    }

    $launchArgs = @()
    if (-not ($ClaudeArgs -contains "--model" -or $ClaudeArgs -contains "-m")) {
        $launchArgs += @("--model", $Model)
    }
    $launchArgs += $ClaudeArgs

    Write-Host "Launching Claude Code with GLM mapping via $BaseUrl"
    Write-Host "Model aliases: sonnet=$SonnetModel, opus=$OpusModel, haiku=$HaikuModel"
    if (-not $NoCredential) {
        Write-Host "Credential source: $(if ($DryRun -and $ApiKey -eq 'dry-run-placeholder') { 'dry-run placeholder' } elseif ($apiKeyWasProvided) { 'ZAI_API_KEY or -ApiKey' } else { 'prompt' })"
    }
    Write-Host "After launch, run /status to confirm routing and /effort to inspect effort."

    if ($DryRun) {
        Write-Host "Dry run: claude $($launchArgs -join ' ')"
    }
    else {
        & $claudeCommand.Source @launchArgs
    }
}
finally {
    if ($apiKeyBstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($apiKeyBstr)
    }
    Restore-ProcessEnv
}
