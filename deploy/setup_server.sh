#!/usr/bin/env bash
# Install or update the shop bot on an Ubuntu server, and run it as two
# background services that start at boot and restart if they crash:
#
#   shop-bot   the WhatsApp webhook server (python whatsapp.py)
#   ngrok      the tunnel that gives it your fixed public address
#
# deploy/upload.ps1 runs this for you. To run it by hand, unpack the project
# into a folder on the server and run:   bash deploy/setup_server.sh
#
# Safe to run again for every update: it replaces the code, but keeps .env,
# the database (holds, handoff tickets) and the ngrok sign-in.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the unpacked upload
RUN_USER="$(id -un)"
APP_DIR="${APP_DIR:-$HOME/shop-bot}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
SHOP_TIMEZONE="${SHOP_TIMEZONE:-Asia/Jerusalem}"
PORT=8000

step() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

step "Copying the code to $APP_DIR"
mkdir -p "$APP_DIR"
if [ "$SOURCE_DIR" != "$APP_DIR" ]; then
    # Everything from the upload except secrets and generated data, which
    # stay as they are on the server.
    (cd "$SOURCE_DIR" && tar --exclude=./.env --exclude=./storage --exclude=./.venv -cf - .) \
        | (cd "$APP_DIR" && tar -xf -)
fi
if [ -f "$HOME/shop-bot.env" ]; then
    mv "$HOME/shop-bot.env" "$APP_DIR/.env"
fi
[ -f "$APP_DIR/.env" ] || fail "No .env found. Copy your .env to $APP_DIR/.env and run this again."
chmod 600 "$APP_DIR/.env"
# Files edited on Windows can have Windows line endings; Linux tools need plain ones.
sed -i 's/\r$//' "$APP_DIR/.env"

env_value() {  # read one setting from .env without printing it
    sed -n "s/^\xEF\xBB\xBF//; s/^$1=//p" "$APP_DIR/.env" | tail -n 1 | sed "s/^[\"']//; s/[\"']$//"
}
for name in OPENAI_API_KEY WHATSAPP_ACCESS_TOKEN WHATSAPP_PHONE_NUMBER_ID WHATSAPP_APP_SECRET WHATSAPP_VERIFY_TOKEN NGROK_AUTHTOKEN; do
    [ -n "$(env_value "$name")" ] || fail "$name is missing from .env. Add it on your computer and upload again."
done

step "Setting the clock to $SHOP_TIMEZONE (hold end times and 'today' use it)"
sudo timedatectl set-timezone "$SHOP_TIMEZONE"

step "Installing system packages (Python tools, ngrok)"
export DEBIAN_FRONTEND=noninteractive
if ! command -v ngrok >/dev/null 2>&1; then
    curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null
fi
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv ngrok >/dev/null
NGROK_BIN="$(command -v ngrok)"

step "Signing ngrok in to your account"
ngrok config add-authtoken "$(env_value NGROK_AUTHTOKEN)" >/dev/null

step "Installing Python packages"
cd "$APP_DIR"
[ -d .venv ] || python3 -m venv ${VENV_OPTIONS:-} .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

step "Building the search index (the database is kept if it exists)"
.venv/bin/python ingest.py --keep-database

step "Installing the services"
sudo tee "$SYSTEMD_DIR/shop-bot.service" >/dev/null <<UNIT
[Unit]
Description=Shop bot WhatsApp webhook server
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$APP_DIR/.venv/bin/python whatsapp.py --port $PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo tee "$SYSTEMD_DIR/shop-bot-ngrok.service" >/dev/null <<UNIT
[Unit]
Description=ngrok tunnel for the shop bot
After=network-online.target shop-bot.service
Wants=network-online.target

[Service]
User=$RUN_USER
ExecStart=$NGROK_BIN http $PORT --log stdout
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable shop-bot.service shop-bot-ngrok.service >/dev/null 2>&1
sudo systemctl restart shop-bot.service shop-bot-ngrok.service

step "Checking that everything answers"
ok=true
for attempt in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then break; fi
    sleep 1
done
if curl -fsS "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
    echo "Bot server:   running"
else
    echo "Bot server:   NOT answering. See: sudo journalctl -u shop-bot -n 50"; ok=false
fi
public_url=""
for attempt in $(seq 1 20); do
    public_url="$(curl -fsS http://127.0.0.1:4040/api/tunnels 2>/dev/null \
        | python3 -c 'import json,sys; t=json.load(sys.stdin)["tunnels"]; print(t[0]["public_url"] if t else "")' 2>/dev/null || true)"
    [ -n "$public_url" ] && break
    sleep 1
done
if [ -n "$public_url" ]; then
    echo "Tunnel:       $public_url"
    if curl -fsS -H "ngrok-skip-browser-warning: 1" "$public_url/" >/dev/null 2>&1; then
        echo "Public check: the bot answers through the tunnel"
    else
        echo "Public check: the tunnel is up but the bot did not answer through it"; ok=false
    fi
    echo
    echo "Meta's Callback URL should be: $public_url/webhook"
else
    echo "Tunnel:       NOT up. If ngrok is still running on your own computer, stop it there;"
    echo "              your address can only be served from one place at a time."
    echo "              Details: sudo journalctl -u shop-bot-ngrok -n 50"
    ok=false
fi

echo
if $ok; then
    echo "Done. The bot now runs on this server and starts by itself after a reboot."
else
    echo "Finished with problems; see the messages above."
    exit 1
fi
