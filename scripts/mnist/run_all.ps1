# Run all MNIST training configurations in sequence and print a comparison table.
#
# Usage:
#   .\run_all.ps1                      # all configs, MlpMatrixNN batch=32 (default)
#   .\run_all.ps1 -NoMatrix            # all configs, classic MlpNN
#   .\run_all.ps1 -BatchSize 64        # override batch size
#   .\run_all.ps1 -Quick               # skip deep-network configs
#   .\run_all.ps1 -Epochs 5            # override epoch count for all runs
#
param(
    [switch] $Quick,
    [switch] $NoMatrix,
    [int]    $Epochs    = 0,   # 0 = use each config's default
    [int]    $BatchSize = 32
)

. "$PSScriptRoot\_common.ps1"

$backend = if ($NoMatrix) { "MlpNN" } else { "MlpMatrixNN batch=$BatchSize" }

# ---------------------------------------------------------------------------
# Config table
# ---------------------------------------------------------------------------
$configs = @(
    [pscustomobject]@{
        Label = "Sigmoid + MSE       (baseline)"
        Act   = "sigmoid"; CE = $false; LR = 0.025; M = 0.9
        HL    = @(300);    E  = 30
    },
    [pscustomobject]@{
        Label = "Sigmoid + CE"
        Act   = "sigmoid"; CE = $true;  LR = 0.05;  M = 0.9
        HL    = @(300);    E  = 30
    },
    [pscustomobject]@{
        Label = "Tanh    + CE"
        Act   = "tanh";    CE = $true;  LR = 0.05;  M = 0.9
        HL    = @(512);    E  = 30
    },
    [pscustomobject]@{
        Label = "ReLU    + MSE"
        Act   = "relu";    CE = $false; LR = 0.01;  M = 0.9
        HL    = @(512);    E  = 30
    },
    [pscustomobject]@{
        Label = "LeakyReLU + CE"
        Act   = "leaky_relu"; CE = $true; LR = 0.01; M = 0.9
        HL    = @(512);    E  = 30
    },
    [pscustomobject]@{
        Label = "Deep Tanh + CE     (512-256-128)"
        Act   = "tanh";    CE = $true;  LR = 0.02;  M = 0.9
        HL    = @(512, 256, 128); E = 50
        Deep  = $true
    },
    [pscustomobject]@{
        Label = "Deep ReLU + CE     (512-256-128)"
        Act   = "relu";    CE = $true;  LR = 0.005; M = 0.9
        HL    = @(512, 256, 128); E = 50
        Deep  = $true
    }
)

if ($Quick) {
    $configs = $configs | Where-Object { -not $_.Deep }
    Write-Host "Quick mode: skipping deep-network configs." -ForegroundColor DarkYellow
}

Write-Host "Backend: $backend" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# Run each config and collect metrics from the log
# ---------------------------------------------------------------------------
$results = @()

foreach ($cfg in $configs) {
    $epochCount = if ($Epochs -gt 0) { $Epochs } else { $cfg.E }
    $costTag    = if ($cfg.CE) { 'ce' } else { 'mse' }
    $hlTag      = $cfg.HL -join '-'
    $backendTag = if ($NoMatrix) { 'mlpnn' } else { "mat_b$BatchSize" }
    $logFile    = "$ModelsDir\$($cfg.Act)_${costTag}_hl${hlTag}_${backendTag}.log"

    $argList = @("-p", $DataPath, "-a", $cfg.Act, "-r", $cfg.LR, "-m", $cfg.M, "-e", $epochCount)
    foreach ($hl in $cfg.HL) { $argList += @("-hl", "$hl") }
    if ($cfg.CE) { $argList += "-c" }

    if ($NoMatrix) {
        # Save model only available for MlpNN
        $modelFile = "$ModelsDir\$($cfg.Act)_${costTag}_hl${hlTag}.json"
        $argList  += @("-s", $modelFile)
    } else {
        $argList += @("-M", "-b", "$BatchSize")
    }

    Run-Training -Label "$($cfg.Label)  [$backend]" -CmdArgs $argList -LogFile $logFile

    $log = Get-Content $logFile -ErrorAction SilentlyContinue

    # Last reported BER (best error rate across all epochs)
    $berLine  = $log | Where-Object { $_ -match "^BER" }        | Select-Object -Last 1
    $ber      = if ($berLine  -match "([\d.]+)%") { [double]$Matches[1] } else { $null }

    # Last reported throughput
    $thruLine = $log | Where-Object { $_ -match "^Throughput" } | Select-Object -Last 1
    $thru     = if ($thruLine -match "([\d]+) samples") { [int]$Matches[1] } else { $null }

    $results += [pscustomobject]@{
        Configuration  = $cfg.Label
        Backend        = $backend
        Cost           = $costTag.ToUpper()
        HiddenLayers   = $hlTag
        Epochs         = $epochCount
        "BER (%)"      = if ($null -ne $ber)  { [math]::Round($ber,  2) } else { "N/A" }
        "Throughput(s/s)" = if ($null -ne $thru) { $thru } else { "N/A" }
    }
}

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 75) -ForegroundColor Cyan
Write-Host "  MNIST Training Results  —  $backend" -ForegroundColor Yellow
Write-Host ("=" * 75) -ForegroundColor Cyan
$results | Sort-Object "BER (%)" | Format-Table -AutoSize
Write-Host "Logs saved to: $ModelsDir" -ForegroundColor DarkGray
