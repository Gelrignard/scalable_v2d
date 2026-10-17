param(
    [ValidateSet('view', 'retarget')][string]$Mode = 'view',
    [int]$MaxFrames = 0,
    [int]$MaxIter = 200,
    [int]$Port = 8080,
    [switch]$Local
)
$ErrorActionPreference = 'Stop'
$v2dRoot = $PSScriptRoot
$v2dMamba = Join-Path $v2dRoot '.tools/Library/bin/micromamba.exe'
$v2dEnv = Join-Path $v2dRoot '.retarget-env'
$v2dScript = Join-Path $v2dRoot 'scripts/local_preview.py'
$v2dArgs = @('run', '-p', $v2dEnv, 'python', '-u', $v2dScript, $Mode, '--port', "$Port", '--max-iter', "$MaxIter")
if ($MaxFrames -gt 0) { $v2dArgs += @('--max-frames', "$MaxFrames") }
if ($Local) { $v2dArgs += '--local' }
& $v2dMamba @v2dArgs
exit $LASTEXITCODE
