param(
    [ValidateSet('run','view','test','evaluate')][string]$Mode = 'view',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$PipelineArgs
)
$ErrorActionPreference = 'Stop'
$v2dMamba = Join-Path $PSScriptRoot '.tools/Library/bin/micromamba.exe'
$v2dEnv = Join-Path $PSScriptRoot '.retarget-env'
if ($Mode -eq 'test') {
    & $v2dMamba run -p $v2dEnv python -m unittest discover -s (Join-Path $PSScriptRoot 'retargeting/tests') -v
} else {
    $v2dScriptName = switch ($Mode) { 'run' {'run.py'} 'view' {'visualize_viser.py'} 'evaluate' {'evaluate.py'} }
    & $v2dMamba run -p $v2dEnv python -u (Join-Path $PSScriptRoot "retargeting/scripts/$v2dScriptName") @PipelineArgs
}
exit $LASTEXITCODE
