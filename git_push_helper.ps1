# Helper: create local git repo and push to GitHub
# Usage:
# 1) Edit $remote to your GitHub repo URL (https or ssh)
# 2) Run in PowerShell from this folder

$remote = 'https://github.com/<your-username>/<repo-name>.git'

if (-not (Test-Path .git)) {
    git init
    git add .
    git commit -m "Initial commit: blackhole_py_demo"
    git branch -M main
    git remote add origin $remote
    git push -u origin main
} else {
    Write-Host "Already a git repo. Use 'git add/commit/push' as needed."
}
