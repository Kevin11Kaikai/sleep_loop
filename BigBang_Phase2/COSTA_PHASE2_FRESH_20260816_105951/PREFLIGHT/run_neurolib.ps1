param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ScriptPath,

    [ValidateRange(1, 3)]
    [int]$RequestedConcurrentProcesses = 1,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ErrorActionPreference = 'Stop'

$campaignRoot = 'D:\Year3_Mao_Projects\sleep_loop\BigBang_Phase2\COSTA_PHASE2_FRESH_20260816_105951'
$condaLauncher = 'C:\Users\YUS190\AppData\Local\anaconda3\condabin\conda.bat'

if (-not (Test-Path -LiteralPath $condaLauncher -PathType Leaf)) {
    throw "Conda launcher unavailable: $condaLauncher"
}

$campaignRootItem = Get-Item -LiteralPath $campaignRoot -Force
$campaignRootResolved = [System.IO.Path]::GetFullPath($campaignRootItem.FullName).TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)
$campaignRootPrefix = $campaignRootResolved + [System.IO.Path]::DirectorySeparatorChar

if (($campaignRootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Campaign root must not be a reparse point: $campaignRootResolved"
}

if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
    throw "Scientific script is not an existing file: $ScriptPath"
}

$resolvedScript = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $ScriptPath).ProviderPath)
if (-not $resolvedScript.StartsWith($campaignRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Scientific scripts must be current-campaign artifacts: $resolvedScript"
}

if ([System.IO.Path]::GetExtension($resolvedScript) -ine '.py') {
    throw "Official scientific scripts must be Python files: $resolvedScript"
}

$pathCursor = Get-Item -LiteralPath $resolvedScript -Force
$rootReached = $false
while ($null -ne $pathCursor) {
    $cursorFullName = [System.IO.Path]::GetFullPath($pathCursor.FullName).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    if (($pathCursor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Reparse points are prohibited in official scientific script paths: $cursorFullName"
    }
    if ($cursorFullName.Equals($campaignRootResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
        $rootReached = $true
        break
    }
    if ($pathCursor -is [System.IO.FileInfo]) {
        $pathCursor = $pathCursor.Directory
    }
    else {
        $pathCursor = $pathCursor.Parent
    }
}

if (-not $rootReached) {
    throw "Scientific script ancestry did not terminate at the campaign root: $resolvedScript"
}

$threadContract = @{
    OMP_NUM_THREADS      = '4'
    MKL_NUM_THREADS      = '4'
    OPENBLAS_NUM_THREADS = '4'
    NUMEXPR_NUM_THREADS  = '4'
}

foreach ($entry in $threadContract.GetEnumerator()) {
    [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
}

foreach ($entry in $threadContract.GetEnumerator()) {
    $observed = [System.Environment]::GetEnvironmentVariable($entry.Key, 'Process')
    if ($observed -ne $entry.Value) {
        throw "Thread contract assertion failed for $($entry.Key): expected $($entry.Value), observed $observed"
    }
}

Add-Type -AssemblyName Microsoft.VisualBasic
$computerInfo = [Microsoft.VisualBasic.Devices.ComputerInfo]::new()
$availableMemory = [uint64]$computerInfo.AvailablePhysicalMemory
$minimumAvailableByConcurrency = @{
    1 = [uint64]6442450944
    2 = [uint64]10737418240
    3 = [uint64]17179869184
}
$requiredMemory = $minimumAvailableByConcurrency[$RequestedConcurrentProcesses]
if ($availableMemory -lt $requiredMemory) {
    throw "Memory admission failed: requested concurrency $RequestedConcurrentProcesses requires $requiredMemory available bytes; observed $availableMemory"
}
[System.Environment]::SetEnvironmentVariable(
    'COSTA_REQUESTED_CONCURRENT_PROCESSES',
    [string]$RequestedConcurrentProcesses,
    'Process'
)

& $condaLauncher run -n neurolib python $resolvedScript @ScriptArgs
exit $LASTEXITCODE
