param(
    [switch]$Worker,
    [string]$LogFile
)

$ErrorActionPreference = 'Stop'
$taskRoot = (Get-Item -LiteralPath $PSScriptRoot).Parent.Parent.FullName
$condaCommand = 'C:\Users\12775\miniconda3\Scripts\conda.exe'
if ($Worker) {
    Set-Location -LiteralPath $taskRoot
    & $condaCommand run --no-capture-output -n polyolefin_ml python -u run_lipo_formal.py --action train --seeds 0 1 *> $LogFile
    exit $LASTEXITCODE
}

$active = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_lipo_formal\.py' }
if ($active) {
    throw "A Lipo formal trainer is already running (PID $($active.ProcessId -join ', '))."
}
$resultRoot = Join-Path (Split-Path -Parent $taskRoot) 'model\results_formal\05_coarse_gnn'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDirectory = Join-Path $resultRoot '_logs'
$backupDirectory = Join-Path $resultRoot "_recovery\before_packed_$stamp"
New-Item -ItemType Directory -Path $logDirectory, $backupDirectory -Force | Out-Null
$trial = Join-Path $resultRoot 'lipo\region_only\tuning\seed_42\trial_0'
foreach ($name in @('resume.pt', 'run_config.json')) {
    $source = Join-Path $trial $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $backupDirectory $name)
    }
}
Copy-Item -LiteralPath (Join-Path $resultRoot 'run_config.json') -Destination (Join-Path $backupDirectory 'root_run_config.json')
$LogFile = Join-Path $logDirectory "lipo_optimized_$stamp.log"
$hostExecutable = (Get-Process -Id $PID).Path
$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $PSCommandPath + '"'),
               '-Worker', '-LogFile', ('"' + $LogFile + '"'))
$process = Start-Process -FilePath $hostExecutable -ArgumentList $arguments -WindowStyle Hidden -PassThru
[pscustomobject]@{ WorkerPid = $process.Id; Log = $LogFile; Backup = $backupDirectory } | ConvertTo-Json
