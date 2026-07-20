<#
Runs the C# WebApi on this machine.

Why not plain `dotnet run`? Windows Smart App Control is enforcing here and blocks
loading AiDirector.WebApi.dll from under Desktop\VideoMaker (FileLoadException
0x800711C7). Publishing to %TEMP% and launching from there is allowed.

The SQLite "Data Source" and the asset dirs in appsettings.json are relative to the
process CWD, so running from the published dir needs absolute overrides - they are
derived from this script's location.

    ./run-local.ps1              # http://localhost:5080
    ./run-local.ps1 -Port 5090
    ./run-local.ps1 -NoPublish   # reuse the last publish, skip the build

NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads BOM-less UTF-8 as
ANSI, and smart dashes/quotes become parser errors.
#>
param(
    [int]$Port = 5080,
    [switch]$NoPublish
)

$ErrorActionPreference = 'Stop'

$csharpDir = $PSScriptRoot
$repoRoot  = Split-Path $csharpDir -Parent          # ...\ai-director
$pubDir    = Join-Path $env:TEMP 'ai-director-csharp-pub'

if (-not $NoPublish) {
    Write-Host "Publishing to $pubDir ..." -ForegroundColor Cyan
    dotnet publish (Join-Path $csharpDir 'src/AiDirector.WebApi') -c Release -o $pubDir --nologo -v q
    if ($LASTEXITCODE -ne 0) { throw "publish failed ($LASTEXITCODE)" }
}

$dll = Join-Path $pubDir 'AiDirector.WebApi.dll'
if (-not (Test-Path $dll)) { throw "$dll not found. Run without -NoPublish first." }

# Absolute paths so the app finds the real DB/assets from the published location.
$env:AiDirector__Paths__Database    = Join-Path $repoRoot 'ai_director.db'
$env:AiDirector__Paths__AssetsDir   = Join-Path $repoRoot 'assets_generated'
$env:AiDirector__Paths__ProjectsDir = Join-Path $repoRoot 'projects'
$env:AiDirector__Paths__ChannelsDir = Join-Path $repoRoot 'channels'

# Keep every clip near 16:9. The inherited default (768x512) is aspect 1.50 while
# the premium opening is 960x544 (1.76), which mixes 3:2 and 16:9 clips inside one
# render. 832x480 (1.73) matches the channel yaml's video_base and is the
# bench-validated VRAM-safe size for 121-frame LTX clips on the 16GB card.
$env:AiDirector__Video__DefaultWidth  = '832'
$env:AiDirector__Video__DefaultHeight = '480'

$contentRoot = Join-Path $csharpDir 'src/AiDirector.WebApi'

Write-Host "DB       : $($env:AiDirector__Paths__Database)"
Write-Host "Listening: http://localhost:$Port  (frontend / , swagger /swagger)" -ForegroundColor Green

& dotnet $dll --urls "http://localhost:$Port" --contentRoot $contentRoot
