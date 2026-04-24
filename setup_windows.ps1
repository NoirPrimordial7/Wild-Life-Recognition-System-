$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/NoirPrimordial7/Wild-life-detection-system.git"
$RepoFolderName = "Wild-life-detection-system"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptRoot

function Write-Step($Message) {
    Write-Host ""
    Write-Host "== $Message =="
}

Write-Step "Checking Git"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed. Install Git from https://git-scm.com/download/win and run this setup again."
    exit 1
}
git --version

Write-Step "Checking Python 3.10"
try {
    & py -3.10 -c "import sys; print(sys.version)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10 check failed."
    }
}
catch {
    Write-Host "Install Python 3.10 from python.org and enable Tcl/Tk and IDLE."
    exit 1
}

$InsideRepo = Test-Path (Join-Path $ProjectRoot "app\ui_wildlife_detector.py")
if (-not $InsideRepo) {
    $TargetRoot = Join-Path $ScriptRoot $RepoFolderName
    if (Test-Path $TargetRoot) {
        Write-Step "Updating existing clone"
        Set-Location $TargetRoot
        git pull --ff-only
    }
    else {
        Write-Step "Cloning project"
        Set-Location $ScriptRoot
        git clone $RepoUrl $RepoFolderName
        Set-Location $TargetRoot
    }
    $ProjectRoot = (Get-Location).Path
}
else {
    Write-Step "Using existing project folder"
    Set-Location $ProjectRoot
    $IsGitRepo = Test-Path (Join-Path $ProjectRoot ".git")
    if ($IsGitRepo) {
        $Status = git status --porcelain
        if ([string]::IsNullOrWhiteSpace($Status)) {
            git pull --ff-only
        }
        else {
            git status --short
            Write-Host "Existing local changes are preserved. Pull manually after committing or stashing local work."
        }
    }
}

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

Write-Step "Creating virtual environment"
if (-not (Test-Path $VenvPython)) {
    & py -3.10 -m venv venv
}
else {
    Write-Host "Reusing existing venv."
}

Write-Step "Installing dependencies"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Step "Running verification"
& $VenvPython app\check_ui_environment.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tkinter/Tcl check failed. Repair Python 3.10 and make sure Tcl/Tk and IDLE are installed."
}
& $VenvPython app\test_model_load.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model test failed."
    exit 1
}

Write-Step "Ready"
Write-Host "Project folder: $ProjectRoot"
Write-Host "Run the UI with:"
Write-Host "  .\venv\Scripts\python.exe app\ui_wildlife_detector.py"
