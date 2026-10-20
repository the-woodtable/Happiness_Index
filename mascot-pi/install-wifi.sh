#!/bin/bash
# Installs the Wi-Fi setup hotspot. Run from ~/mascot with:  sudo bash install-wifi.sh
# Optional: choose the hotspot password:  sudo bash install-wifi.sh MyPassword123
# Safe to run again (keeps your password unless you give a new one).
set -euo pipefail
if [[ $EUID -ne 0 ]]; then echo "Run with sudo:  sudo bash install-wifi.sh"; exit 1; fi
cd "$(dirname "$0")"
[[ -f mascot_wifi.py && -f mascot-wifi.service ]] || { echo "Run this from the folder with mascot_wifi.py"; exit 1; }
systemctl is-active --quiet NetworkManager || { echo "NetworkManager is not running; this needs Bookworm's NetworkManager."; exit 1; }

echo "== Packages"
if ! dpkg -s dnsmasq-base >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends dnsmasq-base   # NetworkManager uses it to hand out addresses on the hotspot
fi

echo "== Settings"
mkdir -p /etc/mascot
MAC=$(cat /sys/class/net/wlan0/address 2>/dev/null | tr -d ':' | tail -c 5 | tr a-z A-Z)
SSID="Mascot-Setup-${MAC:-0000}"
OLD_PW=""
[[ -f /etc/mascot/wifi.env ]] && OLD_PW=$(grep -E '^HOTSPOT_PASSWORD=' /etc/mascot/wifi.env | cut -d= -f2- || true)
PW="${1:-${OLD_PW:-mascot1234}}"
if (( ${#PW} < 8 || ${#PW} > 63 )); then echo "The hotspot password must be 8 to 63 characters."; exit 1; fi
cat > /etc/mascot/wifi.env <<EOF
# Setup hotspot. To change the password: sudo bash install-wifi.sh NewPassword
HOTSPOT_SSID=$SSID
HOTSPOT_PASSWORD=$PW
EOF
chmod 600 /etc/mascot/wifi.env

echo "== Program"
install -d /usr/local/lib/mascot
install -m 755 -o root -g root mascot_wifi.py /usr/local/lib/mascot/mascot_wifi.py

echo "== Hotspot profile '$SSID'"
nmcli connection delete mascot-setup >/dev/null 2>&1 || true
nmcli connection add type wifi ifname wlan0 con-name mascot-setup autoconnect no ssid "$SSID" \
  802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
  ipv4.method shared ipv4.addresses 10.42.0.1/24 ipv6.method disabled \
  wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PW" \
  wifi-sec.proto rsn wifi-sec.pairwise ccmp wifi-sec.group ccmp wifi-sec.pmf disable >/dev/null

echo "== Send every web address to the setup page while the hotspot is on (makes phones pop it up)"
mkdir -p /etc/NetworkManager/dnsmasq-shared.d
echo "address=/#/10.42.0.1" > /etc/NetworkManager/dnsmasq-shared.d/mascot-captive.conf

echo "== Saved Wi-Fi networks: never stop retrying them"
nmcli -t -f NAME,TYPE connection show | while IFS=: read -r name type; do
  if [[ "$type" == "802-11-wireless" && "$name" != "mascot-setup" ]]; then
    nmcli connection modify "$name" connection.autoconnect-retries 0 && echo "   $name"
  fi
done

echo "== Service"
install -m 644 mascot-wifi.service /etc/systemd/system/mascot-wifi.service
systemctl daemon-reload
systemctl enable mascot-wifi.service >/dev/null
systemctl restart mascot-wifi.service
echo
echo "Done. Setup hotspot: $SSID   password: $PW"
echo "It only appears when the Pi has been offline for 5 minutes (90 s after boot)."
echo "Log:  journalctl -u mascot-wifi -f"
