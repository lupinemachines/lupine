param(
    [Parameter(Mandatory = $true)] [string] $CudaVersion,
    [Parameter(Mandatory = $true)] [string] $CudaShortVersion,
    [Parameter(Mandatory = $true)] [string] $CudaSamplesRef,
    [Parameter(Mandatory = $true)] [string] $PytorchIndexUrl,
    [Parameter(Mandatory = $true)] [int] $ServerPortBase,
    [Parameter(Mandatory = $true)] [string] $ResultsDirectory,
    [string] $CudaFlags = '',
    [string] $HostDefines = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CudaShortVersion"
$laneRoot = Join-Path $repoRoot "build\windows-integration-$($CudaShortVersion.Replace('.', '-'))"
$lupineBuild = Join-Path $laneRoot 'lupine'
$serverDirectory = Join-Path $laneRoot 'server'
$cudaSamples = Join-Path $laneRoot 'cuda-samples'
$cudaSamplesBuild = Join-Path $laneRoot 'cuda-samples-build'
$librarySamples = Join-Path $laneRoot 'cuda-library-samples'
$libraryBuild = Join-Path $laneRoot 'cuda-library-build'
$cudaPython = Join-Path $laneRoot 'cuda-python'
$venv = Join-Path $laneRoot 'venv'
$results = [IO.Path]::GetFullPath($ResultsDirectory)
$nvjpegAssets = Join-Path $laneRoot 'nvjpeg-assets'
$serverPort = $ServerPortBase
$script:cudaVersion = $CudaVersion
$script:cases = [Collections.Generic.List[object]]::new()

function Invoke-Checked {
    param([Parameter(Mandatory = $true)] [string] $FilePath,
          [Parameter(ValueFromRemainingArguments = $true)] [string[]] $Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath exited with $LASTEXITCODE"
    }
}

function Get-BashArray {
    param([string] $Path, [string] $Name)
    $text = Get-Content $Path -Raw
    $match = [regex]::Match($text, "(?ms)^$([regex]::Escape($Name))=\(\s*(.*?)^\)")
    if (-not $match.Success) { throw "Could not find $Name in $Path" }
    return [regex]::Matches($match.Groups[1].Value, '(?m)(?:^|\s)([A-Za-z0-9_.+-]+)(?=\s|$)') |
        ForEach-Object { $_.Groups[1].Value }
}

function Find-Executable {
    param([string] $Root, [string] $Name)
    return Get-ChildItem $Root -Recurse -File -Filter "$Name.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\CMakeFiles\\' } |
        Select-Object -First 1
}

function Find-SampleSource {
    param([string] $Name)
    foreach ($group in @('Samples', 'cpp')) {
        $root = Join-Path $cudaSamples $group
        if (Test-Path $root) {
            $directory = Get-ChildItem $root -Directory -Recurse -ErrorAction SilentlyContinue |
                Where-Object Name -eq $Name | Select-Object -First 1
            if ($directory) { return $directory.FullName }
        }
    }
    return $null
}

function Get-SampleArgument {
    param([string] $Name)
    switch ($Name) {
        'FDTD3d' { return @('--qatest') }
        'batchCUBLAS' { return @('-m32', '-n32', '-k32', '-N1') }
        'cuSolverRf' { return @("-file=$(Join-Path (Find-SampleSource 'cuSolverRf') 'lap2D_5pt_n100.mtx')") }
        'bicubicTexture' { return @('-mode=0', '-file=data/0_nearest.ppm') }
        'bilateralFilter' { return @('-radius=5', '-file=data/ref_05.ppm') }
        'bindlessTexture' { return @('-file=data/ref_bindlessTexture.bin') }
        'boxFilter' { return @('-radius=14', '-file=data/ref_14.ppm') }
        'eigenvalues' { return @('-matrix-size=128', '-iters-timing=1') }
        'FunctionPointers' { return @('-mode=0', '-file=data/ref_orig.pgm') }
        'imageDenoising' { return @('-kernel=0', '-file=data/ref_passthru.ppm') }
        'Mandelbrot' { return @('-mode=0', '-file=data/Mandelbrot_fp32.ppm') }
        'marchingCubes' { return @('-dump=0', '-file=data/posArray.bin') }
        { $_ -in @('matrixMul', 'matrixMul_nvrtc') } { return @('-wA=32', '-hA=32', '-wB=32', '-hB=32') }
        'matrixMulCUBLAS' { return @('-sizemult=1') }
        'nbody' { return @('-benchmark', '-numbodies=4096', '-i=1') }
        'NV12toBGRandResize' { return @('-input=data/test640x480.nv12', '-width=640', '-height=480', '-dst_width=320', '-dst_height=240', '-batch=1') }
        'oceanFFT' { return @('-qatest') }
        'particles' { return @('-file=data/ref_particles.bin') }
        'ptxgen' { return @('test.ll') }
        'randomFog' { return @('-qatest') }
        { $_ -in @('reduction', 'threadFenceReduction') } { return @('-n=1024', '-threads=64', '-maxblocks=16') }
        'recursiveGaussian' { return @('-benchmark') }
        'simpleGL' { return @('-file=data/ref_simpleGL.bin') }
        'smokeParticles' { return @('-qatest') }
        'SobelFilter' { return @('-mode=0', '-file=data/ref_orig.pgm') }
        'simpleTexture3D' { return @('-file=data/ref_texture3D.bin') }
        'transpose' { return @('-dimX=512', '-dimY=512') }
        'UnifiedMemoryPerf' { return @('-kernel-iterations=1') }
        'volumeFiltering' { return @('-file=data/ref_volumefilter.ppm') }
        'volumeRender' { return @('--file=ref_volume.ppm') }
        default { return @() }
    }
}

function Get-SampleWorkingDirectory {
    param([string] $Name, [string] $ExecutableDirectory)
    $sourceWorkingDirectory = @(
        'bicubicTexture', 'bilateralFilter', 'bindlessTexture', 'boxFilter',
        'FunctionPointers', 'imageDenoising', 'Mandelbrot', 'marchingCubes',
        'nbody', 'NV12toBGRandResize', 'oceanFFT', 'particles', 'ptxgen',
        'randomFog', 'recursiveGaussian', 'simpleGL', 'smokeParticles',
        'SobelFilter', 'simpleTexture3D', 'volumeFiltering', 'volumeRender'
    )
    if ($Name -in $sourceWorkingDirectory) { return Find-SampleSource $Name }
    return $ExecutableDirectory
}

function Wait-TcpPort {
    param([int] $Port)
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $client = [Net.Sockets.TcpClient]::new()
            $client.Connect('127.0.0.1', $Port)
            $client.Dispose()
            return
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    throw "Server did not listen on port $Port"
}

function Convert-ToArgumentString {
    param([string[]] $Arguments)
    return ($Arguments | ForEach-Object {
        if ($_ -notmatch '[\s"]') { $_ } else { '"' + $_.Replace('"', '\"') + '"' }
    }) -join ' '
}

function Invoke-IsolatedCase {
    param(
        [string] $Suite,
        [string] $Name,
        [string] $FilePath,
        [string[]] $Arguments = @(),
        [string] $WorkingDirectory = '',
        [int] $TimeoutSeconds = 900,
        [int[]] $SkipExitCodes = @(),
        [string] $SkipReason = '',
        [string] $FailureReason = ''
    )

    $caseStart = [DateTime]::UtcNow
    $caseDirectory = Join-Path $results "cases\$Suite\$($Name.Replace('/', '_').Replace('\', '_'))"
    New-Item -ItemType Directory -Force $caseDirectory | Out-Null
    $stdout = Join-Path $caseDirectory 'stdout.log'
    $stderr = Join-Path $caseDirectory 'stderr.log'
    $serverLog = Join-Path $caseDirectory 'server.log'
    $status = 'passed'
    $message = ''

    if ($FailureReason) {
        $status = 'failed'
        $message = $FailureReason
    } elseif ($SkipReason) {
        $status = 'skipped'
        $message = $SkipReason
    } else {
        $port = $script:serverPort
        $script:serverPort += 1
        $previousPort = $env:LUPINE_PORT
        $previousServer = $env:LUPINE_SERVER
        $previousDisableLocal = $env:LUPINE_DISABLE_LOCAL
        $env:LUPINE_PORT = [string]$port
        $server = $null
        try {
            $server = Start-Process $script:serverExe -PassThru -NoNewWindow `
                -RedirectStandardOutput $serverLog -RedirectStandardError "$serverLog.stderr"
            Wait-TcpPort $port
            $env:LUPINE_SERVER = "127.0.0.1:$port"
            $env:LUPINE_DISABLE_LOCAL = '1'
            if (-not $WorkingDirectory) { $WorkingDirectory = Split-Path $FilePath }
            $process = Start-Process $FilePath -ArgumentList (Convert-ToArgumentString $Arguments) `
                -WorkingDirectory $WorkingDirectory -PassThru -NoNewWindow `
                -RedirectStandardOutput $stdout -RedirectStandardError $stderr
            if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
                $process.Kill()
                $status = 'failed'
                $message = "Timed out after $TimeoutSeconds seconds"
            } elseif ($process.ExitCode -in $SkipExitCodes) {
                $status = 'skipped'
                $message = "Exited with expected skip code $($process.ExitCode)"
            } elseif ($process.ExitCode -ne 0) {
                $status = 'failed'
                $message = "Exited with $($process.ExitCode)"
            }
        } catch {
            $status = 'failed'
            $message = $_.Exception.Message
        } finally {
            if ($null -ne $server -and -not $server.HasExited) { $server.Kill() }
            $env:LUPINE_PORT = $previousPort
            $env:LUPINE_SERVER = $previousServer
            $env:LUPINE_DISABLE_LOCAL = $previousDisableLocal
        }
    }

    $duration = ([DateTime]::UtcNow - $caseStart).TotalSeconds
    $script:cases.Add([pscustomobject]@{
        Suite = $Suite; Name = $Name; Status = $status; Message = $message; Duration = $duration
    })
    Write-Output "$status $Suite.$Name ($([Math]::Round($duration, 2))s) $message"
}

function Write-Junit {
    $settings = [Xml.XmlWriterSettings]::new()
    $settings.Indent = $true
    $writer = [Xml.XmlWriter]::Create((Join-Path $results 'junit.xml'), $settings)
    try {
        $failed = @($script:cases | Where-Object Status -eq 'failed').Count
        $skipped = @($script:cases | Where-Object Status -eq 'skipped').Count
        $writer.WriteStartDocument()
        $writer.WriteStartElement('testsuite')
        $writer.WriteAttributeString('name', "Windows CUDA $script:cudaVersion")
        $writer.WriteAttributeString('tests', [string]$script:cases.Count)
        $writer.WriteAttributeString('failures', [string]$failed)
        $writer.WriteAttributeString('skipped', [string]$skipped)
        foreach ($case in $script:cases) {
            $writer.WriteStartElement('testcase')
            $writer.WriteAttributeString('classname', $case.Suite)
            $writer.WriteAttributeString('name', $case.Name)
            $writer.WriteAttributeString('time', $case.Duration.ToString('0.000', [Globalization.CultureInfo]::InvariantCulture))
            if ($case.Status -eq 'failed') {
                $writer.WriteStartElement('failure'); $writer.WriteString($case.Message); $writer.WriteEndElement()
            } elseif ($case.Status -eq 'skipped') {
                $writer.WriteStartElement('skipped'); $writer.WriteAttributeString('message', $case.Message); $writer.WriteEndElement()
            }
            $writer.WriteEndElement()
        }
        $writer.WriteEndElement()
        $writer.WriteEndDocument()
    } finally {
        $writer.Dispose()
    }
}

if (-not (Test-Path "$cudaRoot\bin\nvcc.exe")) { throw "CUDA toolkit is missing: $cudaRoot" }
New-Item -ItemType Directory -Force $laneRoot, $serverDirectory, $results, `
    (Join-Path $nvjpegAssets 'decoded'), (Join-Path $nvjpegAssets 'resized'), `
    (Join-Path $nvjpegAssets 'watermarked') | Out-Null
$env:CUDA_PATH = $cudaRoot
$env:Path = "$cudaRoot\bin;C:\Program Files\Git\cmd;C:\Program Files\CMake\bin;C:\ProgramData\chocolatey\bin;$env:Path"
$env:CL = $HostDefines
$env:CUDA_BINDINGS_SKIP_EXAMPLE = '100'

$cmakeArgs = @(
    '-S', $repoRoot, '-B', $lupineBuild, '-G', 'Visual Studio 17 2022', '-A', 'x64',
    '-T', "cuda=$cudaRoot",
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake',
    '-DVCPKG_TARGET_TRIPLET=x64-windows-static', '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded',
    '-DOPENSSL_USE_STATIC_LIBS=ON', "-DCMAKE_CUDA_FLAGS=$CudaFlags", "-DCUDAToolkit_ROOT=$cudaRoot"
)
Invoke-Checked cmake @cmakeArgs
Invoke-Checked cmake --build $lupineBuild --config Release --parallel `
    --target lupine_driver_server lupine_cuda_client lupine_nvml_client
$builtServer = Join-Path $lupineBuild 'Release\lupine_driver_server.exe'
$script:serverExe = Join-Path $serverDirectory 'lupine_driver_server.exe'
$clientDll = Join-Path $lupineBuild 'Release\nvcuda.dll'
$nvmlDll = Join-Path $lupineBuild 'Release\nvml.dll'
foreach ($path in @($builtServer, $clientDll, $nvmlDll)) {
    if (-not (Test-Path $path)) { throw "Missing Lupine artifact: $path" }
}
Copy-Item $builtServer $script:serverExe -Force

if (-not (Test-Path "$cudaSamples\.git")) {
    Invoke-Checked git clone --branch $CudaSamplesRef --depth 1 https://github.com/NVIDIA/cuda-samples.git $cudaSamples
}
if (Test-Path "$cudaSamples\CMakeLists.txt") {
    Invoke-Checked cmake -S $cudaSamples -B $cudaSamplesBuild -A x64 -T "cuda=$cudaRoot" `
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
        -DENABLE_CUDA_C_LINKING_SAMPLE=1 "-DCUDAToolkit_ROOT=$cudaRoot"
    try {
        Invoke-Checked cmake --build $cudaSamplesBuild --config Release --parallel
    } catch {
        Write-Warning "The CUDA Samples aggregate build had failures; produced samples will still run: $_"
    }
} else {
    $solution = Get-ChildItem $cudaSamples -Filter 'Samples_vs*.sln' | Sort-Object Name -Descending | Select-Object -First 1
    if ($null -eq $solution) { throw "No CUDA Samples build entry point for $CudaSamplesRef" }
    $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    $vsRoot = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
    $msbuild = Join-Path $vsRoot 'MSBuild\Current\Bin\MSBuild.exe'
    try {
        Invoke-Checked $msbuild $solution.FullName /m /p:Configuration=Release /p:Platform=x64
    } catch {
        Write-Warning "The CUDA Samples aggregate build had failures; produced samples will still run: $_"
    }
}

$sampleNames = @(
    Get-BashArray (Join-Path $repoRoot 'test\run_cuda_samples.sh') 'CORE_SAMPLES'
    Get-BashArray (Join-Path $repoRoot 'test\run_cuda_samples.sh') 'LIBRARY_SAMPLES'
)
foreach ($sample in $sampleNames) {
    $exeName = if ($sample -eq 'device-side-launch') { 'dsl' } else { $sample }
    $exe = Find-Executable $cudaSamplesBuild $exeName
    if ($null -eq $exe) { $exe = Find-Executable $cudaSamples $exeName }
    if ($null -eq $exe) {
        Invoke-IsolatedCase 'cuda' $sample '' -SkipReason 'build did not produce an executable'
        continue
    }
    Copy-Item $clientDll (Join-Path $exe.DirectoryName 'nvcuda.dll') -Force
    Copy-Item $nvmlDll (Join-Path $exe.DirectoryName 'nvml.dll') -Force
    if ($sample -like '*_nvrtc') {
        foreach ($includeDirectory in @('nv', 'cuda')) {
            $source = Join-Path "$cudaRoot\include" $includeDirectory
            $destination = Join-Path $exe.DirectoryName $includeDirectory
            if ((Test-Path $source) -and -not (Test-Path $destination)) {
                Copy-Item $source $destination -Recurse
            }
        }
    }
    $arguments = Get-SampleArgument $sample
    $workingDirectory = Get-SampleWorkingDirectory $sample $exe.DirectoryName
    $timeout = if ($sample -in @(
        'simpleStreams', 'scan', 'LargeKernelParameter', 'UnifiedMemoryStreams',
        'UnifiedMemoryPerf', 'HSOpticalFlow', 'jacobiCudaGraphs',
        'radixSortThrust', 'segmentationTreeThrust', 'batchCUBLAS', 'cuSolverRf',
        'conjugateGradientPrecond', 'watershedSegmentationNPP')) { 600 } else { 120 }
    Invoke-IsolatedCase -Suite 'cuda' -Name $sample -FilePath $exe.FullName `
        -Arguments $arguments -WorkingDirectory $workingDirectory `
        -TimeoutSeconds $timeout -SkipExitCodes @(2)
}

if (-not (Test-Path "$librarySamples\.git")) {
    Invoke-Checked git clone https://github.com/NVIDIA/CUDALibrarySamples.git $librarySamples
    Invoke-Checked git -C $librarySamples checkout 3437729
}
$libraryRoots = @('cuBLAS', 'cuBLASLt', 'cuFFT', 'cuRAND', 'cuSOLVER', 'cuSPARSE', 'nvJPEG', 'NPP')
$allProjects = @(Get-ChildItem -Path ($libraryRoots | ForEach-Object { Join-Path $librarySamples $_ }) `
    -Recurse -File -Filter CMakeLists.txt -ErrorAction SilentlyContinue)
$projects = $allProjects | Where-Object {
    $prefix = $_.Directory.FullName.TrimEnd('\') + '\'
    -not ($allProjects | Where-Object { $_.Directory.FullName.StartsWith($prefix) })
} | Sort-Object FullName
$knownFailures = Get-Content (Join-Path $repoRoot 'test\cuda-library-samples\known_failures.txt') |
    ForEach-Object { ($_ -split '#', 2)[0].Trim() } | Where-Object { $_ }
foreach ($project in $projects) {
    $relative = [IO.Path]::GetRelativePath($librarySamples, $project.Directory.FullName)
    $build = Join-Path $libraryBuild $relative
    $built = $true
    try {
        Invoke-Checked cmake -S $project.Directory.FullName -B $build -A x64 `
            -T "cuda=$cudaRoot" `
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
            "-DCUDAToolkit_ROOT=$cudaRoot"
        Invoke-Checked cmake --build $build --config Release --parallel 2
    } catch {
        $built = $false
    }
    $executables = if ($built) {
        @(Get-ChildItem $build -Recurse -File -Filter '*.exe' |
            Where-Object FullName -notmatch '\\CMakeFiles\\')
    } else { @() }
    if ($executables.Count -eq 0) {
        Invoke-IsolatedCase 'cuda-library' $relative '' -SkipReason 'build did not produce an executable'
        continue
    }
    foreach ($exe in $executables) {
        $unit = "$($relative.Replace('\', '/'))/$($exe.BaseName)"
        $knownUnit = $knownFailures | Where-Object {
            $_ -eq $unit -or $_ -eq "$($relative.Replace('\', '/'))/bin/$($exe.BaseName)"
        }
        if ($knownUnit -or $unit -match '^cuFFT/3d_mgpu_') {
            Invoke-IsolatedCase -Suite 'cuda-library' -Name $unit -FilePath '' `
                -SkipReason 'requires unsupported hardware or is a tracked native failure'
            continue
        }
        Copy-Item $clientDll (Join-Path $exe.DirectoryName 'nvcuda.dll') -Force
        Copy-Item $nvmlDll (Join-Path $exe.DirectoryName 'nvml.dll') -Force
        $images = Join-Path $librarySamples 'nvJPEG\nvJPEG-Decoder\input_images'
        $arguments = switch -Wildcard ($unit) {
            'nvJPEG/nvJPEG-Decoder/*' { @('-i', $images, '-b', '2', '-o', (Join-Path $nvjpegAssets 'decoded')); break }
            'nvJPEG/nvJPEG-Decoder-Backend-ROI/*' { @('-i', $images, '-b', '2'); break }
            'nvJPEG/nvJPEG-Decoder-MultipleInstances/*' { @('-i', $images, '-s', '2', '-j', '2', '-r', '1'); break }
            'nvJPEG/nvJPEG-Encoder-MultipleInstances/*' { @('-i', (Join-Path $nvjpegAssets 'decoded'), '-s', '2', '-j', '2', '-r', '1'); break }
            'nvJPEG/Image-Resize/*' { @('-i', $images, '-o', (Join-Path $nvjpegAssets 'resized')); break }
            'nvJPEG/Image-Resize-WaterMark/*' { @('-i', (Join-Path $librarySamples 'nvJPEG\Image-Resize-WaterMark\input_images'), '-o', (Join-Path $nvjpegAssets 'watermarked')); break }
            'NPP/nppCanny/*' { @('example_input.png'); break }
            default { @() }
        }
        $workingDirectory = switch -Wildcard ($unit) {
            'NPP/findContour/*' { $project.Directory.FullName; break }
            'NPP/nppCanny/*' { $project.Directory.FullName; break }
            'nvJPEG/Image-Resize-WaterMark/*' { $project.Directory.FullName; break }
            'NPP/*' { Join-Path $project.Directory.FullName 'images'; break }
            default { $exe.DirectoryName }
        }
        Invoke-IsolatedCase -Suite 'cuda-library' -Name $unit -FilePath $exe.FullName `
            -Arguments $arguments -WorkingDirectory $workingDirectory
    }
}

if (-not (Test-Path "$venv\Scripts\python.exe")) { Invoke-Checked python -m venv $venv }
$python = Join-Path $venv 'Scripts\python.exe'
Invoke-Checked $python -m pip install --disable-pip-version-check --upgrade pip
Invoke-Checked $python -m pip install --disable-pip-version-check numpy pytest pytest-benchmark
Invoke-Checked $python -m pip install --disable-pip-version-check --index-url $PytorchIndexUrl torch
$cudaPackage = if ($CudaShortVersion -eq '11.8') { 'cuda-python==11.8.*' } else { "cuda-bindings==$($CudaShortVersion.Split('.')[0]).*" }
Invoke-Checked $python -m pip install --disable-pip-version-check $cudaPackage
Copy-Item $clientDll (Join-Path $venv 'Scripts\nvcuda.dll') -Force
Copy-Item $nvmlDll (Join-Path $venv 'Scripts\nvml.dll') -Force

$pytorchTests = @('discover', 'tensor_ops', 'matmul', 'fft', 'cudnn_conv', 'sparse_mm',
    'linalg_solve', 'autograd_step', 'compile_elementwise', 'microgpt_train')
foreach ($test in $pytorchTests) {
    Invoke-IsolatedCase -Suite 'pytorch' -Name $test -FilePath $python `
        -Arguments @((Join-Path $repoRoot 'test\pytorch_lupine_tests.py'), $test) `
        -WorkingDirectory $repoRoot
}

$packageName = if ($CudaShortVersion -eq '11.8') { 'cuda-python' } else { 'cuda-bindings' }
$installedVersion = & $python -c "from importlib.metadata import version; print(version('$packageName'))"
if (-not (Test-Path "$cudaPython\.git")) {
    Invoke-Checked git clone https://github.com/NVIDIA/cuda-python.git $cudaPython
}
Invoke-Checked git -C $cudaPython fetch --tags origin
Invoke-Checked git -C $cudaPython checkout "v$installedVersion"
$testsDirectory = @('cuda_bindings\tests', 'cuda\tests') |
    ForEach-Object { Join-Path $cudaPython $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1
$examplesDirectory = @('cuda_bindings\examples', 'examples') |
    ForEach-Object { Join-Path $cudaPython $_ } | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $testsDirectory) { throw "No test directory found in cuda-python $installedVersion" }
Get-ChildItem $testsDirectory -File -Filter 'test_*.py' |
    Where-Object Name -notmatch '^test_(cufile|examples|nvvm|nvjitlink|nvfatbin)\.py$' |
    ForEach-Object {
        Invoke-IsolatedCase -Suite 'cuda-python' -Name $_.Name -FilePath $python `
            -Arguments @('-m', 'pytest', '-q', '--no-header', '-p', 'no:cacheprovider', $_.FullName) `
            -WorkingDirectory $testsDirectory -SkipExitCodes @(5, 100)
    }
if ($examplesDirectory) {
    $exampleFilter = if (Get-ChildItem $examplesDirectory -Recurse -File -Filter '*_test.py' |
        Select-Object -First 1) { '*_test.py' } else { '*.py' }
    Get-ChildItem $examplesDirectory -Recurse -File -Filter $exampleFilter |
        Where-Object { $_.Name -notin @('conftest.py', '__init__.py') -and $_.Name -notlike 'numba_*' -and $_.FullName -notmatch '\\common\\' } |
        ForEach-Object {
            $relative = [IO.Path]::GetRelativePath($examplesDirectory, $_.FullName)
            $arguments = if ($_.Name -like '*_test.py') {
                @('-m', 'pytest', '-q', '--no-header', '-p', 'no:cacheprovider', $_.FullName)
            } else { @($_.FullName) }
            Invoke-IsolatedCase -Suite 'cuda-python' -Name $relative -FilePath $python `
                -Arguments $arguments -WorkingDirectory $examplesDirectory `
                -SkipExitCodes @(5, 100)
        }
}

$customBuild = Join-Path $laneRoot 'custom'
New-Item -ItemType Directory -Force $customBuild | Out-Null
foreach ($source in Get-ChildItem (Join-Path $repoRoot 'test') -File -Filter 'test_*.cu') {
    $exe = Join-Path $customBuild "$($source.BaseName).exe"
    try {
        $nvccArguments = @('--cudart=shared', '-Wno-deprecated-gpu-targets', '-arch=sm_89')
        if ($CudaFlags) { $nvccArguments += $CudaFlags }
        $nvccArguments += @($source.FullName, '-o', $exe, '-lcuda', '-lcublas')
        Invoke-Checked "$cudaRoot\bin\nvcc.exe" @nvccArguments
        Copy-Item $clientDll (Join-Path $customBuild 'nvcuda.dll') -Force
        Copy-Item $nvmlDll (Join-Path $customBuild 'nvml.dll') -Force
        Invoke-IsolatedCase -Suite 'custom' -Name $source.BaseName -FilePath $exe `
            -WorkingDirectory $customBuild
    } catch {
        Invoke-IsolatedCase -Suite 'custom' -Name $source.BaseName -FilePath '' `
            -FailureReason "build failed: $($_.Exception.Message)"
    }
}

Write-Junit
$failures = @($script:cases | Where-Object Status -eq 'failed')
if ($failures.Count -ne 0) { throw "$($failures.Count) Windows integration tests failed" }
