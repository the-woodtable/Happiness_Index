#!/bin/bash
# Installs the mascot meter as a system service. Run from ~/mascot with:  sudo bash install.sh
# Safe to run again after an update (it never overwrites your settings files).
set -euo pipefail

APP=/home/mascot/mascot
if [[ $EUID -ne 0 ]]; then echo "Run with sudo:  sudo bash install.sh"; exit 1; fi
if ! id mascot >/dev/null 2>&1; then echo "User 'mascot' not found. This script expects the user 'mascot'."; exit 1; fi
cd "$APP"

echo "== System libraries MediaPipe needs (missing on Raspberry Pi OS Lite)"
if ! dpkg -s libegl1 libgles2 >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends libegl1 libgles2
fi

echo "== Checking files"
for f in mascot_meter.py mascot_cloud.py mascot.service face_landmarker.task venv/bin/python; do
  [[ -e "$f" ]] || { echo "MISSING: $APP/$f"; exit 1; }
done
SZ=$(stat -c %s face_landmarker.task)
(( SZ > 1000000 )) || { echo "face_landmarker.task looks broken ($SZ bytes)"; exit 1; }
mkdir -p "$APP/data"
chown -R mascot:mascot "$APP/data" "$APP"/*.py

echo "== Settings in /etc/mascot"
mkdir -p /etc/mascot
if [[ ! -f /etc/mascot/firebase.env ]]; then
  cat > /etc/mascot/firebase.env <<'EOF'
# Firebase settings for the Pi (same values you typed into pi_cloud_test.py).
# No quotes, no spaces around the = sign.
FIREBASE_API_KEY=PASTE_API_KEY
FIREBASE_DB_URL=https://PASTE_PROJECT_ID-default-rtdb.asia-southeast1.firebasedatabase.app
FIREBASE_EMAIL=PASTE_PI_ACCOUNT_EMAIL
FIREBASE_PASSWORD=PASTE_PI_ACCOUNT_PASSWORD
EOF
  echo "   created /etc/mascot/firebase.env (needs your values)"
fi
chown root:mascot /etc/mascot/firebase.env
chmod 640 /etc/mascot/firebase.env
if [[ ! -f /etc/mascot/meter.env ]]; then
  cat > /etc/mascot/meter.env <<'EOF'
# Options for mascot_meter.py. Edit, then: sudo systemctl restart mascot
# Examples: --size 640x360   --fps 15   --min-face-frac 0.12   --rotate 180   --tune
MASCOT_ARGS=--size 640x480 --fps 12
EOF
  echo "   created /etc/mascot/meter.env"
fi

echo "== Hardware watchdog (reboots the Pi if the whole system freezes)"
mkdir -p /etc/systemd/system.conf.d
cat > /etc/systemd/system.conf.d/10-mascot-watchdog.conf <<'EOF'
[Manager]
RuntimeWatchdogSec=15
RebootWatchdogSec=2min
EOF

echo "== Persistent, size-capped logs"
mkdir -p /etc/systemd/journald.conf.d /var/log/journal
cat > /etc/systemd/journald.conf.d/10-mascot.conf <<'EOF'
[Journal]
Storage=persistent
SystemMaxUse=200M
EOF
systemctl restart systemd-journald

echo "== Service"
install -m 644 mascot.service /etc/systemd/system/mascot.service
systemctl daemon-reload
systemctl daemon-reexec   # applies the watchdog setting now
systemctl enable mascot.service >/dev/null

if grep -q PASTE /etc/mascot/firebase.env; then
  echo
  echo "Installed, NOT started yet: fill in /etc/mascot/firebase.env first:"
  echo "   sudo nano /etc/mascot/firebase.env"
  echo "then:  sudo systemctl restart mascot"
else
  systemctl restart mascot.service
  echo
  echo "Installed and started. Watch the log with:  journalctl -u mascot -f"
fi
