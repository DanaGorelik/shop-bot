<#
Upload the project to your server and install or update it there.

Run from the project folder:

    .\deploy\upload.ps1 -Server ubuntu@203.0.113.10 -Key C:\keys\shop-bot.key

-Server  "ubuntu@" followed by the server's public IP address (from Oracle's console)
-Key     the private key file you downloaded when creating the server

What it does:
  1. Restricts the key file's permissions (ssh refuses keys others can read).
  2. Packs the code and shop data into one file. .env, .venv and storage are
     never packed.
  3. Copies that file, and your .env separately, to the server.
  4. Runs deploy/setup_server.sh there, which installs or updates everything
     and restarts the bot.

Run it again whenever you change something: holds, handoff tickets and the
ngrok sign-in on the server are kept.
#>
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [Parameter(Mandatory = $true)][string]$Key
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path (Join-Path $root ".env"))) { throw "No .env found in $root" }
if (-not (Test-Path $Key)) { throw "Key file not found: $Key" }
if ($Server -notmatch "@") { throw "-Server should look like ubuntu@203.0.113.10" }

Write-Host "Restricting the key file to your user..."
icacls $Key /inheritance:r | Out-Null
icacls $Key /grant:r "${env:USERNAME}:(R)" | Out-Null

$bundle = Join-Path $env:TEMP "shop-bot-upload.tgz"
if (Test-Path $bundle) { Remove-Item $bundle }
Write-Host "Packing the project (without .env, .venv or storage)..."
$items = @("shopbot", "data", "evals", "deploy", "tests", "chat.py", "ingest.py", "search.py",
           "whatsapp.py", "evaluate.py", "requirements.txt", "README.md")
tar.exe -czf $bundle --exclude=__pycache__ --exclude=evals/results -C $root $items
if ($LASTEXITCODE -ne 0) { throw "Packing failed" }

Write-Host "Copying to $Server..."
scp -i $Key $bundle "${Server}:shop-bot-upload.tgz"
if ($LASTEXITCODE -ne 0) { throw "Copying the project failed (see the message above)" }
scp -i $Key (Join-Path $root ".env") "${Server}:shop-bot.env"
if ($LASTEXITCODE -ne 0) { throw "Copying .env failed" }

Write-Host "Installing on the server (the first time takes a few minutes)..."
$remote = 'rm -rf ~/shop-bot-incoming && mkdir ~/shop-bot-incoming && ' +
          'tar -xzf ~/shop-bot-upload.tgz -C ~/shop-bot-incoming && ' +
          'sed -i ''s/\r$//'' ~/shop-bot-incoming/deploy/setup_server.sh && ' +
          'bash ~/shop-bot-incoming/deploy/setup_server.sh'
ssh -i $Key $Server $remote
if ($LASTEXITCODE -ne 0) { throw "Setup on the server reported a problem (see above)" }
