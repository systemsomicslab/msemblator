# Creates "MS-Emblator" shortcuts that start msemblator_gui.cmd with the MS-Emblator icon.
#
#   scripts\create_shortcut.cmd                  desktop and Start menu
#   scripts\create_shortcut.cmd -Desktop         desktop only
#   scripts\create_shortcut.cmd -StartMenu       Start menu only
#   scripts\create_shortcut.cmd -Folder D:\x     into the given folder(s)
#
# The shortcut points to this copy of the repository; after moving the repository,
# create it again. Prints the path of every shortcut it wrote.
param(
    [switch]$Desktop,
    [switch]$StartMenu,
    [string[]]$Folder
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root 'msemblator_gui.cmd'
$icon = Join-Path $root 'src\msemblator\gui\assets\msemblator.ico'
foreach ($required in @($target, $icon)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Not found: $required" }
}

$folders = @()
if ($Folder) {
    $folders = $Folder
} else {
    if ($Desktop -or -not $StartMenu) { $folders += [Environment]::GetFolderPath('Desktop') }
    if ($StartMenu -or -not $Desktop) { $folders += [Environment]::GetFolderPath('Programs') }
}

$shell = New-Object -ComObject WScript.Shell
foreach ($place in $folders) {
    if (-not (Test-Path -LiteralPath $place)) { New-Item -ItemType Directory -Path $place | Out-Null }
    $path = Join-Path $place 'MS-Emblator.lnk'
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = $target
    $shortcut.WorkingDirectory = $root
    $shortcut.IconLocation = "$icon,0"
    $shortcut.Description = 'MS-Emblator: ensemble annotation of MS/MS spectra'
    $shortcut.WindowStyle = 1   # normal window: the first start shows the environment set-up
    $shortcut.Save()
    Write-Output $path
}
