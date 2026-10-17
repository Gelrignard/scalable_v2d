param(
    [ValidateSet('demo', 'test', 'fetch', 'diagnostic', 'book', 'view-book')]
    [string]$Mode = 'demo',
    [double]$MassKg,
    [int]$Episode = 11,
    [int]$Window = 11
)
$ErrorActionPreference = 'Stop'
$taskWorkspace = Split-Path -Parent $PSScriptRoot
$taskMamba = Join-Path $taskWorkspace '.tools/Library/bin/micromamba.exe'
$taskEnvironment = Join-Path $taskWorkspace '.retarget-env'
if (-not (Test-Path -LiteralPath $taskMamba)) {
    throw 'Existing local runtime not found. See correction/README.md for dependencies.'
}
switch ($Mode) {
    'demo' { $taskPythonArgs = @('-m', 'correction.demo') }
    'book' { $taskPythonArgs = @('-m', 'correction.book_pipeline') }
    'view-book' { $taskPythonArgs = @('-m', 'correction.view_book') }
    'test' { $taskPythonArgs = @('-m', 'unittest', 'discover', '-s', 'correction/tests', '-v') }
    'fetch' { $taskPythonArgs = @('-m', 'correction.fetch_sample', '--episode', "$Episode") }
    'diagnostic' {
        if (-not $PSBoundParameters.ContainsKey('MassKg')) {
            throw 'Supply -MassKg explicitly: mass is a pilot assumption, not a dataset measurement.'
        }
        $taskMassText = $MassKg.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        $taskOutput = 'correction/outputs/tier1_episode_{0:D6}' -f $Episode
        $taskPythonArgs = @('-m', 'correction.tier1_diagnostic', '--episode', "$Episode", '--mass-kg', $taskMassText,
                            '--window', "$Window", '--output', $taskOutput)
    }
}
Push-Location -LiteralPath $taskWorkspace
try {
    & $taskMamba run -p $taskEnvironment python @taskPythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Correction command failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
