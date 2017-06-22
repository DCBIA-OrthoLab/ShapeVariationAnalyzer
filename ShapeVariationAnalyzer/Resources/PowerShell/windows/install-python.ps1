trap { Write-Error $_; Exit 1 }

#
# By default, Python 2.7.12, 3.5.3 and 3.6.1 are installed.
#
# Setting $pythonVersion to "2.7", "3.4", "3.5" or "3.6" allows to install a specific version
#
# Setting $pythonArch to either "64" or "86" allows to install python for specific architecture.
#
# Setting $pythonPrependPath to 1 will add install and Scripts directories the PATH and .PY to PATHEXT. The variable
# should be set only if $pythonVersion and $pythonArch are set. By default, the value is 0.
#

if (![System.IO.File]::Exists(".\install-utils.ps1")) {
  Write-Host "Download install-utils.ps1"
  $url = "https://raw.githubusercontent.com/DCBIA-OrthoLab/ShapeVariationAnalyzer/release/src/ShapeVariationAnalyzer/Resources/PowerShell/windows/install-utils.ps1"
  $cwd = (Get-Item -Path ".\" -Verbose).FullName
  (new-object net.webclient).DownloadFile($url, "$cwd\install-utils.ps1")
}
Import-Module .\install-utils.ps1 -Force

function Get-Python-InstallPath {
param (
  [string]$pythonVersion,
  [string]$pythonArch
  )
  $suffix = '-32'
  if ($pythonArch.CompareTo('64') -eq 0) {
    $suffix = ''
  }
  $roots = @("HKCU", "HKLM")
  foreach ($root in $roots) {
    $path = "$($root):\Software\Python\PythonCore\$pythonVersion$suffix\InstallPath"
    if (Test-Path -Path $path -PathType Container) {
      $properties = Get-ItemProperty -Path $path
      if ($properties -And (Get-Member -InputObject $properties -Name '(Default)')) {
        $installPath = (Get-ItemProperty -Path $path -Name '(Default)').'(Default)'
        return (Resolve-Path(Join-Path $installPath "\\")).Path
      }
    }
  }
  return ""
}

function Install-Python {
param (
  [string]$installerPath,
  [string]$targetDir
  )

  Write-Host "Installing $installerPath into $targetDir"
  $interpreter = Join-Path $targetDir "python.exe"
  if ([System.IO.Directory]::Exists($interpreter)) {
    Write-Host "-> skipping: found $interpreter"
  return
  }
  if (![System.IO.Directory]::Exists($targetDir)) {
    [System.IO.Directory]::CreateDirectory($targetDir)
  }
  #
  # See https://docs.python.org/3.6/using/windows.html#installing-without-ui
  #
  Start-Process $installerPath -ArgumentList "TargetDir=$targetDir DefaultAllUsersTargetDir=$targetDir InstallAllUsers=1 Include_launcher=0 PrependPath=$pythonPrependPath Shortcuts=0 /passive" -NoNewWindow -Wait
}

function Install-Python-27-33-34 {
param (
  [string]$targetDir,
  [string]$installerName,
  [string]$downloadURL
  )
  Download-URL $downloadURL $downloadDir
  Install-MSI $installerName $downloadDir $targetDir
  Install-Pip $targetDir $downloadDir
  Pip-Install $targetDir 'virtualenv'
  if ($pythonPrependPath -eq 1) {
    Write-Host "Pre-pending '$targetDir;$targetDir\Scripts\' to PATH"
    [Environment]::SetEnvironmentVariable("Path", "$targetDir;$targetDir\Scripts\;$env:Path", "Machine")
  }
}

# See https://pip.pypa.io/en/stable/installing/
function Install-Pip {
param (
  [string]$pythonDir,
  [string]$downloadDir
  )
  Download-URL 'https://bootstrap.pypa.io/get-pip.py' $downloadDir

  $get_pip_script = Join-Path $downloadDir "get-pip.py"

  $interpreter = Join-Path $pythonDir "python.exe"
  Write-Host "Installing pip using $interpreter"

  Start-Process $interpreter -ArgumentList "`"$get_pip_script`"" -NoNewWindow -Wait
}

function Pip-Install {
param (
  [string]$pythonDir,
  [string]$package
  )

  $pip = Join-Path $pythonDir "Scripts\\pip.exe"

  Write-Host "Installing $package using $pip"

  Start-Process $pip -ArgumentList "install `"$package`"" -NoNewWindow -Wait
}

$downloadDir = "C:/Downloads"

if ($pythonVersion) {
  $pythonVersion = [string]::Join("", $pythonVersion.Split("."), 0, 2)
  Write-Host "Installing Python version $pythonVersion"
}

if ($pythonArch) {
  if(!($pythonArch -match "^(64|86)$")){
    throw "'pythonArch' variable incorrectly set to [$pythonArch]. Hint: '64' or '86' value is expected."
  }
  Write-Host "Installing Python for architecture x$pythonArch"
}

if (!$pythonVersion -Or !$pythonArch) {
  if ($pythonPrependPath) {
    throw "'pythonPrependPath' variable should explicitly be set when both 'pythonVersion' and 'pythonArch' are set"
  }
}
if (!$pythonPrependPath) {
  $pythonPrependPath = 0
  Write-Host "Defaulting 'pythonPrependPath' variable to 0."
}

if(!($pythonPrependPath -match "^(0|1)$")){
  throw "'$pythonPrependPath' variable incorrectly set to [$pythonPrependPath]. Hint: '0' or '1' value is expected."
}


$exeVersions = @("2.7.12", "3.3.5", "3.4.4")
foreach ($version in $exeVersions) {

  $split = $version.Split(".")
  $majorMinor = [string]::Join("", $split, 0, 2)
  $majorMinorDot = [string]::Join(".", $split, 0, 2)

  if($pythonVersion -And ! $pythonVersion.CompareTo($majorMinor) -eq 0) {
    Write-Host "Skipping $majorMinor"
    continue
  }

  if (!$pythonArch -Or $pythonArch.CompareTo("64") -eq 0) {
    $targetDir = "C:\Python$($majorMinor)-x64"
    $installerName = "python-$($version).amd64.msi"
    $downloadURL = "https://www.python.org/ftp/python/$($version)/$($installerName)"
    Install-Python-27-33-34 $targetDir $installerName $downloadURL
  }

  if (!$pythonArch -Or $pythonArch.CompareTo("86") -eq 0) {
    $targetDir = "C:\Python$($majorMinor)-x86"
    $installerName = "python-$($version).msi"
    $downloadURL = "https://www.python.org/ftp/python/$($version)/$($installerName)"
    Install-Python-27-33-34 $targetDir $installerName $downloadURL
  }
}

$exeVersions = @("3.5.3", "3.6.1")
foreach ($version in $exeVersions) {

  $split = $version.Split(".")
  $majorMinor = [string]::Join("", $split, 0, 2)
  $majorMinorDot = [string]::Join(".", $split, 0, 2)

  if($pythonVersion -And ! $pythonVersion.CompareTo($majorMinor) -eq 0) {
    Write-Host "Skipping $majorMinor"
    continue
  }

  if (!$pythonArch -Or $pythonArch.CompareTo("64") -eq 0) {

    Download-URL "https://www.python.org/ftp/python/$($version)/python-$($version)-amd64.exe" $downloadDir

    $pythonInstallPath = Get-Python-InstallPath $majorMinorDot "64"
    $targetInstallPath = "C:\Python$($majorMinor)-x64\"
    $installerPath = Join-Path $downloadDir "python-$($version)-amd64.exe"

    if (!$pythonInstallPath.CompareTo($targetInstallPath) -eq 0) {
      if ($pythonInstallPath) {
        Write-Host "Found a python installation in a different directory [$pythonInstallPath] - Uninstalling"
        Start-Process $installerPath -ArgumentList "/uninstall /passive" -NoNewWindow -Wait
      }
    } elseif ($pythonInstallPath) {
      Write-Host "Updating existing installation [$pythonInstallPath]"
    }

    Install-Python $installerPath $targetInstallPath
    Install-Pip $targetInstallPath $downloadDir
    Pip-Install $targetInstallPath 'virtualenv'
  }

  if (!$pythonArch -Or $pythonArch.CompareTo("86") -eq 0) {
    Download-URL "https://www.python.org/ftp/python/$($version)/python-$($version).exe" $downloadDir

    $pythonInstallPath = Get-Python-InstallPath $majorMinorDot "86"
    $targetInstallPath = "C:\Python$($majorMinor)-x86\"
    $installerPath = Join-Path $downloadDir "python-$($version).exe"

    if (!$pythonInstallPath.CompareTo($targetInstallPath) -eq 0) {
      if ($pythonInstallPath) {
        Write-Host "Found a python installation in a different directory [$pythonInstallPath] - Uninstalling"
        Start-Process $installerPath -ArgumentList "/uninstall /passive" -NoNewWindow -Wait
      }
    } elseif ($pythonInstallPath) {
      Write-Host "Updating existing installation [$pythonInstallPath]"
    }

    Install-Python  $installerPath $targetInstallPath
    Install-Pip $targetInstallPath $downloadDir
    Pip-Install $targetInstallPath 'virtualenv'
  }
  
}
