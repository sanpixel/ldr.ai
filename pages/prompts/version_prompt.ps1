# Prompt Versioning Helper Script
# Usage: .\version_prompt.ps1 -version "1.1" -description "Added mixed content handling"

param(
    [Parameter(Mandatory=$true)]
    [string]$version,
    
    [Parameter(Mandatory=$true)]
    [string]$description
)

# Paths
$rootPath = Split-Path -Path $PSScriptRoot -Parent | Split-Path -Parent
$promptFile = Join-Path $rootPath "bearings_prompt.txt"
$backupFile = Join-Path $PSScriptRoot "bearings_prompt_v$version.txt"
$readmeFile = Join-Path $PSScriptRoot "README.md"

Write-Host "📦 Creating backup of current prompt..." -ForegroundColor Yellow

# Check if current prompt exists
if (-not (Test-Path $promptFile)) {
    Write-Host "❌ Error: bearings_prompt.txt not found in root directory" -ForegroundColor Red
    exit 1
}

# Create backup
try {
    Copy-Item $promptFile $backupFile
    Write-Host "✅ Backed up current prompt to: bearings_prompt_v$version.txt" -ForegroundColor Green
} catch {
    Write-Host "❌ Error creating backup: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Get current date
$date = Get-Date -Format "yyyy-MM-dd"

Write-Host "📝 Updating version documentation..." -ForegroundColor Yellow

# Update README with new version info
$newVersionEntry = @"

### v$version ($date)
**$description**

**Status:** Active
"@

# Read current README
$readmeContent = Get-Content $readmeFile -Raw

# Update current version section
$readmeContent = $readmeContent -replace "## Current Version: v[\d\.]+", "## Current Version: v$version"
$readmeContent = $readmeContent -replace "(\*\*Status:\*\*) Active", "`$1 Superseded"

# Add new version entry after the version history header
$readmeContent = $readmeContent -replace "(## Version History)", "`$1$newVersionEntry"

# Write back to file
Set-Content $readmeFile $readmeContent -Encoding UTF8

Write-Host "✅ Updated README.md with version v$version" -ForegroundColor Green

Write-Host @"
🎉 Prompt versioning complete!

Next steps:
1. Edit bearings_prompt.txt with your changes
2. Test the new prompt using the reasoning dashboard
3. Commit changes:
   git add .
   git commit -m "Update prompt to v$version: $description"

To rollback if needed:
   copy pages\prompts\bearings_prompt_v$version.txt bearings_prompt.txt
"@ -ForegroundColor Cyan
