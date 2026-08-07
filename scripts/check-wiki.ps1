[CmdletBinding()]
param(
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRootPath = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$wikiRoot = Join-Path $repositoryRootPath 'wiki'

if (-not (Test-Path -LiteralPath $wikiRoot -PathType Container)) {
    throw "Wiki directory not found: $wikiRoot"
}

$errors = [System.Collections.Generic.List[string]]::new()
$pages = @(Get-ChildItem -LiteralPath $wikiRoot -Filter '*.md' -File)
$pageNames = @{}

foreach ($page in $pages) {
    $pageNames[$page.BaseName.ToLowerInvariant()] = $true
}

foreach ($page in $pages) {
    $text = Get-Content -LiteralPath $page.FullName -Raw
    $relativeName = "wiki/$($page.Name)"

    $fenceCount = [regex]::Matches($text, '(?m)^```').Count
    if (($fenceCount % 2) -ne 0) {
        $errors.Add("$($relativeName): unbalanced fenced code blocks")
    }

    if ($page.Name -ne '_Sidebar.md') {
        if ($text -notmatch '(?m)^# [^#]') {
            $errors.Add("$($relativeName): missing level-one title")
        }

        if ($text -notmatch '(?m)^## Keep reading\s*$') {
            $errors.Add("$($relativeName): missing 'Keep reading' section")
        }
    }

    if ($text -match 'https://github\.com/eantcal/nunn/(?:blob|tree)/master(?:/|$)') {
        $errors.Add("$($relativeName): source link still targets the obsolete master branch")
    }

    $linkText = [regex]::Replace($text, '(?ms)^```.*?^```\s*', '')
    $links = [regex]::Matches($linkText, '\]\((?<target>[^)\s]+)')
    foreach ($link in $links) {
        $target = $link.Groups['target'].Value.Trim('<', '>')

        if ($target.StartsWith('#')) {
            continue
        }

        if ($target -match '^https://github\.com/eantcal/nunn/(?:blob|tree)/main/(?<path>[^#?]+)') {
            $sourcePath = [Uri]::UnescapeDataString($Matches['path']).Replace('/', [IO.Path]::DirectorySeparatorChar)
            $localSource = Join-Path $repositoryRootPath $sourcePath
            if (-not (Test-Path -LiteralPath $localSource)) {
                $errors.Add("$($relativeName): missing linked repository path '$sourcePath'")
            }
            continue
        }

        if ($target -match '^[a-z][a-z0-9+.-]*://') {
            continue
        }

        $pathOnly = ($target -split '#', 2)[0]
        if ([string]::IsNullOrWhiteSpace($pathOnly)) {
            continue
        }

        $decoded = [Uri]::UnescapeDataString($pathOnly)
        if ($decoded.StartsWith('assets/')) {
            $assetPath = Join-Path $wikiRoot $decoded.Replace('/', [IO.Path]::DirectorySeparatorChar)
            if (-not (Test-Path -LiteralPath $assetPath -PathType Leaf)) {
                $errors.Add("$($relativeName): missing asset '$decoded'")
            }
            continue
        }

        if ($decoded.Contains('/')) {
            $relativePath = Join-Path $wikiRoot $decoded.Replace('/', [IO.Path]::DirectorySeparatorChar)
            if (-not (Test-Path -LiteralPath $relativePath)) {
                $errors.Add("$($relativeName): missing relative target '$decoded'")
            }
            continue
        }

        $pageTarget = [IO.Path]::GetFileNameWithoutExtension($decoded).ToLowerInvariant()
        if (-not $pageNames.ContainsKey($pageTarget)) {
            $errors.Add("$($relativeName): missing wiki page '$decoded'")
        }
    }
}

if ($errors.Count -gt 0) {
    Write-Error ("Wiki validation failed:`n - " + ($errors -join "`n - "))
    exit 1
}

$assetCount = @(Get-ChildItem -LiteralPath (Join-Path $wikiRoot 'assets') -File).Count
Write-Host "Wiki validation passed: $($pages.Count) pages, $assetCount assets."
