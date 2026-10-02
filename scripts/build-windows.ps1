param(
    [string]$TensorRTRoot = $env:TENSORRT_ROOT,
    [string]$OpenCVDir = $env:OpenCV_DIR,
    [string]$BuildDir = 'build'
)

$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$localCmake = [System.IO.Path]::GetFullPath((Join-Path $projectRoot '..\toolchains\vs2022-buildtools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'))

if (Test-Path -LiteralPath $localCmake -PathType Leaf) {
    $cmake = $localCmake
} else {
    $command = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $command) {
        throw 'CMake not found. Install Visual Studio Build Tools with CMake, or put CMake on PATH.'
    }
    $cmake = $command.Source
}

if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $buildDir = [System.IO.Path]::GetFullPath($BuildDir)
} else {
    $buildDir = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $BuildDir))
}
if ([string]::IsNullOrWhiteSpace($TensorRTRoot)) {
    throw 'TensorRT root is required. Pass -TensorRTRoot C:\path\to\TensorRT-10 or set TENSORRT_ROOT.'
}
$configureArgs = @('-S', $projectRoot, '-B', $buildDir,
                   '-G', 'Visual Studio 17 2022', '-A', 'x64',
                   "-DTENSORRT_ROOT=$TensorRTRoot")
if (-not [string]::IsNullOrWhiteSpace($OpenCVDir)) {
    $configureArgs += "-DOpenCV_DIR=$OpenCVDir"
}
& $cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed with exit code $LASTEXITCODE" }

& $cmake --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }

Write-Host "Built: $(Join-Path $buildDir 'Release\video_infer.exe')"
