# Run all MNIST training configurations in sequence and print a comparison table.
#
# Usage:
#   .\run_all.ps1              # runs all configs
#   .\run_all.ps1 -Quick       # shallow configs only (no deep networks)
#   .\run_all.ps1 -Epochs 10   # override epoch count for all runs
#
param(
    [switch] $Quick,
    [int]    $Epochs = 0   # 0 = use each script's default
)

. "$PSScriptRoot\_common.ps1"

# ---------------------------------------------------------------------------
# Config table: [label, activation, cost_flag, lr, momentum, hidden_layers[], epochs]
# ---------------------------------------------------------------------------
$configs = @(
    [pscustomobject]@{
        Label   = "Sigmoid + MSE       (baseline)"
        Act     = "sigmoid"; CE = $false; LR = 0.025; M = 0.9
        HL      = @(300);    E  = 30
    },
    [pscustomobject]@{
        Label   = "Sigmoid + CE"
        Act     = "sigmoid"; CE = $true;  LR = 0.05;  M = 0.9
        HL      = @(300);    E  = 30
    },
    [pscustomobject]@{
        Label   = "Tanh    + CE"
        Act     = "tanh";    CE = $true;  LR = 0.05;  M = 0.9
        HL      = @(512);    E  = 30
    },
    [pscustomobject]@{
        Label   = "ReLU    + MSE"
        Act     = "relu";    CE = $false; LR = 0.01;  M = 0.9
        HL      = @(512);    E  = 30
    },
    [pscustomobject]@{
        Label   = "LeakyReLU + CE"
        Act     = "leaky_relu"; CE = $true; LR = 0.01; M = 0.9
        HL      = @(512);    E  = 30
    },
    [pscustomobject]@{
        Label   = "Deep Tanh + CE     (512-256-128)"
        Act     = "tanh";    CE = $true;  LR = 0.02;  M = 0.9
        HL      = @(512, 256, 128); E = 50
        Deep    = $true
    },
    [pscustomobject]@{
        Label   = "Deep ReLU + CE     (512-256-128)"
        Act     = "relu";    CE = $true;  LR = 0.005; M = 0.9
        HL      = @(512, 256, 128); E = 50
        Deep    = $true
    }
)

if ($Quick) {
    $configs = $configs | Where-Object { -not $_.Deep }
    Write-Host "Quick mode: skipping deep-network configs." -ForegroundColor DarkYellow
}

# ---------------------------------------------------------------------------
# Run each config and collect last-epoch BER from stdout
# ---------------------------------------------------------------------------
$results = @()

foreach ($cfg in $configs) {
    $epochCount = if ($Epochs -gt 0) { $Epochs } else { $cfg.E }
    $modelFile  = "$ModelsDir\$($cfg.Act)_$($if($cfg.CE){'ce'}else{'mse'})_hl$($cfg.HL -join '-').json"
    $logFile    = $modelFile -replace '\.json$', '.log'

    $argList = @("-p", $DataPath, "-a", $cfg.Act, "-r", $cfg.LR, "-m", $cfg.M, "-e", $epochCount)
    foreach ($hl in $cfg.HL) { $argList += @("-hl", "$hl") }
    if ($cfg.CE) { $argList += "-c" }
    $argList += @("-s", $modelFile)

    $output = Run-Training -Label $cfg.Label -Args $argList -LogFile $logFile

    # Extract best error rate (BER) from the last "BER" line in the log.
    $berLine = (Get-Content $logFile -ErrorAction SilentlyContinue) |
               Where-Object { $_ -match "^BER" } | Select-Object -Last 1
    $ber = if ($berLine -match "([\d.]+)%") { [double]$Matches[1] } else { $null }

    $results += [pscustomobject]@{
        Configuration = $cfg.Label
        Activation    = $cfg.Act
        Cost          = if ($cfg.CE) { "CrossEntropy" } else { "MSE" }
        HiddenLayers  = $cfg.HL -join "-"
        Epochs        = $epochCount
        "BER (%)"     = if ($null -ne $ber) { [math]::Round($ber, 2) } else { "N/A" }
    }
}

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  MNIST Training Results" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor Cyan
$results | Sort-Object "BER (%)" | Format-Table -AutoSize
Write-Host "Models saved to: $ModelsDir" -ForegroundColor DarkGray
