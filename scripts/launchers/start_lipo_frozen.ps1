param([switch]$Worker, [string]$LogFile)
$ErrorActionPreference = 'Stop'
$taskRoot = (Get-Item -LiteralPath $PSScriptRoot).Parent.Parent.FullName
if ($Worker) {
    Set-Location -LiteralPath $taskRoot
    & 'C:\Users\12775\miniconda3\Scripts\conda.exe' run --no-capture-output -n polyolefin_ml python -u run_lipo_frozen.py --action train --seeds 0 1 *> $LogFile
    exit $LASTEXITCODE
}
$active = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'run_lipo_(formal|frozen)\.py' }
if ($active) { throw "A Lipo process is already running (PID $($active.ProcessId -join ', '))." }
$resultRoot = Join-Path (Split-Path -Parent $taskRoot) 'model\results_formal\05_coarse_gnn\frozen_mechanism'
$logDirectory = Join-Path $resultRoot '_logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$LogFile = Join-Path $logDirectory ('lipo_frozen_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
$hostExecutable = (Get-Process -Id $PID).Path
$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $PSCommandPath + '"'),
               '-Worker', '-LogFile', ('"' + $LogFile + '"'))
$process = Start-Process -FilePath $hostExecutable -ArgumentList $arguments -WindowStyle Hidden -PassThru
[pscustomobject]@{WorkerPid=$process.Id; Log=$LogFile; Results=$resultRoot} | ConvertTo-Json
