trap { Write-Error $_; Exit 1 }

$scriptName = "install-python.ps1"
if (![System.IO.File]::Exists(".\windows\$scriptName")) {
  Write-Host "Download $scriptName"
  $url = "https://raw.githubusercontent.com/pdedumast/ShapeVariationAnalyzer/master/src/ShapeVariationAnalyzer/Resources/PowerShell/windows/$scriptName"
  $cwd = (Get-Item -Path ".\" -Verbose).FullName
  (new-object net.webclient).DownloadFile($url, "$cwd\$scriptName")
}

$pythonPrependPath = "1"
$pythonVersion = "3.5"
$pythonArch = "64"
Invoke-Expression ".\$scriptName"
