#!/usr/bin/env python3
"""
Wi-Fi setup hotspot for the mascot Pi (Raspberry Pi OS Bookworm, NetworkManager). Runs as root.

Normal mode
  Checks the internet every 15 s. NetworkManager connects to saved Wi-Fi and Ethernet by itself
  (Ethernet wins automatically when plugged in).
  Offline for 5 min (or 90 s right after boot) -> setup mode.

Setup mode
  The Pi becomes the Wi-Fi network "Mascot-Setup-XXXX" (password in /etc/mascot/wifi.env).
  Join it with a phone or laptop: the setup page pops up (or open http://10.42.0.1). Pick a network,
  type its password, tap Connect.
    - success: the hotspot disappears and the Pi is back online
    - failure (wrong password etc.): the hotspot comes back in about a minute with the error shown
  While nobody is using the page, every 5 min it drops the hotspot for ~45 s to retry saved
  networks, so it also recovers by itself when the venue Wi-Fi comes back.
  Plugging in Ethernet ends setup mode within 15 s.

Force setup mode now:   sudo touch /run/mascot-wifi-setup
Logs:                   journalctl -u mascot-wifi -f

Does not work with: WPA2/WPA3-Enterprise networks (that ask for a username as well as a password,
e.g. eduroam, many offices) and networks that need a login/accept web page (hotels, malls).
"""
from __future__ import annotations

import html
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENV_FILE = "/etc/mascot/wifi.env"
TRIGGER = "/run/mascot-wifi-setup"
IFACE = "wlan0"
HOTSPOT_CON = "mascot-setup"
AP_IP = "10.42.0.1"

CHECK_EVERY_S = 15
OFFLINE_TO_SETUP_S = 300      # offline this long -> setup mode
BOOT_GRACE_S = 90             # right after boot, don't wait the full 5 min
RETRY_EVERY_S = 300           # in setup mode, retry saved networks this often...
IDLE_BEFORE_RETRY_S = 120     # ...but only if nobody has used the setup page for this long
RETRY_FOR_S = 45
CONNECT_WAIT_S = 45
CHECK_URLS = (
    ("http://connectivitycheck.gstatic.com/generate_204", 204, None),
    ("http://www.msftconnecttest.com/connecttest.txt", 200, b"Microsoft Connect Test"),
)


def log(msg):
    print(msg, flush=True)


# ---- system helpers (replaceable in tests) -------------------------------------
def nmcli(*args, timeout=60):
    try:
        p = subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "nmcli timed out"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None   # a redirect means a login page is in the way: treat as offline


def internet_ok() -> bool:
    opener = urllib.request.build_opener(_NoRedirect)
    for url, code, body in CHECK_URLS:
        try:
            with opener.open(url, timeout=5) as r:
                if r.status == code and (body is None or r.read(200).startswith(body)):
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return False


def split_terse(line: str) -> list:
    """Split one line of `nmcli -t` output (':' separated, '\\:' and '\\\\' escaped)."""
    out, cur, i = [], [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            cur.append(line[i + 1])
            i += 2
            continue
        if c == ":":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def read_env(path=ENV_FILE) -> dict:
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return vals


# ---- the setup page ----------------------------------------------------------------
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Mascot Wi-Fi setup</title>
<style>
body{{margin:0;font:16px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#4f3a97;background:#fff}}
header{{background:#4f3a97;color:#fff;padding:14px 16px;border-bottom:4px solid #fed200}}
h1{{margin:0;font-size:20px}} main{{padding:12px 16px;max-width:520px;margin:0 auto}}
.msg{{background:#fed200;color:#2b2050;padding:10px 12px;border-radius:8px;font-weight:600;margin:8px 0}}
.err{{background:#ffe1e1;color:#8a1020}}
label.net{{display:flex;align-items:center;gap:10px;padding:12px 10px;border:2px solid #ece4f4;border-radius:10px;margin:8px 0}}
label.net input{{width:20px;height:20px}} .ssid{{flex:1;font-weight:700;word-break:break-all}} .meta{{font-size:13px;white-space:nowrap}}
input[type=text],input[type=password]{{width:100%;box-sizing:border-box;font:inherit;padding:10px;border:2px solid #4f3a97;border-radius:8px}}
button{{width:100%;font:inherit;font-weight:700;background:#4f3a97;color:#fff;border:0;border-radius:10px;padding:14px;margin-top:14px}}
.small{{font-size:13px}} a{{color:#4f3a97}}
</style></head><body><header><h1>Mascot Wi-Fi setup</h1></header><main>
{message}
<form method="post" action="/connect">
<p><b>1. Choose the Wi-Fi network</b></p>
{networks}
<label class="net"><input type="radio" name="ssid" value="__other__" {other_checked}><span class="ssid">Other / hidden network</span></label>
<input type="text" name="ssid_other" placeholder="Network name (exactly as written)" autocapitalize="off" autocorrect="off">
<p><b>2. Password</b></p>
<input type="password" name="password" id="pw" placeholder="Leave empty for an open network" autocapitalize="off" autocorrect="off">
<p class="small"><label><input type="checkbox" onclick="pw.type=this.checked?'text':'password'"> Show password</label></p>
<button type="submit">Connect</button>
</form>
<p class="small"><a href="/?rescan=1">Network not listed? Refresh the list</a> (the hotspot drops for about 20 seconds; rejoin it).</p>
<p class="small">Won't work: networks that ask for a username as well as a password (offices, universities), or that need a
login/"accept" web page (hotels, malls). Use a phone hotspot or ask for a normal password network instead.</p>
<p class="small">{status}</p>
</main></body></html>"""

DONE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Connecting</title>
<style>body{{margin:0;font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#4f3a97}}
header{{background:#4f3a97;color:#fff;padding:14px 16px;border-bottom:4px solid #fed200}}h1{{margin:0;font-size:20px}}
main{{padding:12px 16px;max-width:520px;margin:0 auto}}</style></head><body><header><h1>{title}</h1></header><main>
{body}</main></body></html>"""


def signal_bars(sig: int) -> str:
    return "▂▄▆█"[: max(1, min(4, sig // 25 + 1))]


class Portal:
    """Shared state between the web page (its own threads) and the manager loop."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.networks = []          # [(ssid, signal, secured)]
        self.message = ""
        self.error = False
        self.pending = None         # (ssid, password, hidden) waiting for the manager
        self.rescan_requested = False
        self.last_hit = 0.0
        self.offline_since_txt = ""

    def render(self) -> str:
        with self.lock:
            nets, msg, err = list(self.networks), self.message, self.error
            since = self.offline_since_txt
        rows = []
        for i, (ssid, sig, secured) in enumerate(nets):
            e = html.escape(ssid, quote=True)
            rows.append(f'<label class="net"><input type="radio" name="ssid" value="{e}" {"checked" if i == 0 else ""}>'
                        f'<span class="ssid">{e}</span><span class="meta">{"🔒 " if secured else "open "}'
                        f'{signal_bars(sig)}</span></label>')
        if not rows:
            rows.append('<p class="small">No networks found. Use "Other" below, or refresh the list.</p>')
        message = f'<div class="msg{" err" if err else ""}">{html.escape(msg)}</div>' if msg else ""
        return PAGE.format(message=message, networks="\n".join(rows),
                           other_checked="" if nets else "checked",
                           status=html.escape(f"Offline since {since}." if since else ""))


def make_handler(portal: Portal):
    class Handler(BaseHTTPRequestHandler):
        server_version = "mascot-setup"

        def log_message(self, *a):
            pass

        def _send(self, code, body: str, extra=None):
            b = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _redirect(self):
            # Phones probe a known URL when joining Wi-Fi. Anything other than the expected reply makes
            # them show the "sign in to network" popup, which opens this page.
            self._send(302, "", {"Location": f"http://{AP_IP}/"})

        def do_GET(self):
            portal.last_hit = portal.clock()
            u = urllib.parse.urlparse(self.path)
            host = (self.headers.get("Host") or "").split(":")[0]
            if u.path != "/" or host not in (AP_IP, ""):
                return self._redirect()
            if "rescan=1" in u.query:
                portal.rescan_requested = True
                return self._send(200, DONE_PAGE.format(
                    title="Refreshing the list",
                    body="<p>The hotspot will disappear for about 20 seconds while the Pi looks for networks.</p>"
                         "<p>Then rejoin <b>the setup hotspot</b> and the page will open again.</p>"))
            self._send(200, portal.render())

        def do_POST(self):
            portal.last_hit = portal.clock()
            if urllib.parse.urlparse(self.path).path != "/connect":
                return self._redirect()
            n = min(int(self.headers.get("Content-Length") or 0), 4096)
            form = urllib.parse.parse_qs(self.rfile.read(n).decode(errors="replace"), keep_blank_values=True)
            ssid = (form.get("ssid") or [""])[0]
            hidden = False
            if ssid == "__other__" or not ssid:
                ssid, hidden = (form.get("ssid_other") or [""])[0].strip(), True
            pw = (form.get("password") or [""])[0]
            if not ssid:
                with portal.lock:
                    portal.message, portal.error = "Choose a network first.", True
                return self._send(303, "", {"Location": "/"})
            if pw and not (8 <= len(pw) <= 63):
                with portal.lock:
                    portal.message, portal.error = "Wi-Fi passwords are 8 to 63 characters. Check and try again.", True
                return self._send(303, "", {"Location": "/"})
            with portal.lock:
                portal.pending = (ssid, pw, hidden)
            e = html.escape(ssid)
            self._send(200, DONE_PAGE.format(
                title="Connecting…",
                body=f"<p>The Pi is now connecting to <b>{e}</b>. This setup hotspot will disappear.</p>"
                     "<p><b>If it worked:</b> the hotspot stays gone and the website shows <b>Live</b> within about a "
                     "minute. You can rejoin your normal Wi-Fi.</p>"
                     "<p><b>If it didn't</b> (wrong password, network out of range): the setup hotspot comes back in "
                     "about a minute. Rejoin it and the page will show what went wrong.</p>"))
    return Handler


# ---- the manager -----------------------------------------------------------------
class WifiManager:
    def __init__(self, nm=nmcli, online=internet_ok, port=80, clock=time.monotonic, sleep=time.sleep):
        self.nm, self.online, self.port, self.clock, self.sleep = nm, online, port, clock, sleep
        self.portal = Portal(clock)
        self.server = None
        self.mode = "normal"
        self.start = clock()
        self.ever_online = False
        self.offline_since = clock()
        self.last_retry = 0.0
        env = read_env()
        self.hotspot_name = env.get("HOTSPOT_SSID", "Mascot-Setup")

    # -- radio
    def scan(self):
        code, out = self.nm("-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list",
                            "ifname", IFACE, "--rescan", "yes", timeout=30)
        best = {}
        if code == 0:
            for line in out.splitlines():
                parts = split_terse(line)
                if len(parts) < 3 or not parts[0] or parts[0] == self.hotspot_name:
                    continue
                try:
                    sig = int(parts[1])
                except ValueError:
                    sig = 0
                secured = parts[2] not in ("", "--")
                if parts[0] not in best or sig > best[parts[0]][1]:
                    best[parts[0]] = (parts[0], sig, secured)
        else:
            log(f"[wifi] scan failed: {out}")
        nets = sorted(best.values(), key=lambda n: -n[1])[:25]
        with self.portal.lock:
            self.portal.networks = nets
        log(f"[wifi] found {len(nets)} networks: {', '.join(n[0] for n in nets[:8])}")

    def hotspot_up(self):
        code, out = self.nm("connection", "up", HOTSPOT_CON, timeout=45)
        if code != 0:
            log(f"[wifi] could not start hotspot: {out}")
            return False
        log(f"[wifi] setup hotspot '{self.hotspot_name}' is ON (page: http://{AP_IP})")
        return True

    def hotspot_down(self):
        self.nm("connection", "down", HOTSPOT_CON, timeout=30)

    def wait_online(self, seconds) -> bool:
        end = self.clock() + seconds
        while self.clock() < end:
            self.sleep(5)
            if self.online():
                return True
        return False

    # -- web page
    def server_start(self):
        if self.server is not None:
            return
        ThreadingHTTPServer.allow_reuse_address = True
        try:
            self.server = ThreadingHTTPServer(("0.0.0.0", self.port), make_handler(self.portal))
        except OSError as e:
            log(f"[wifi] could not start the setup page on port {self.port}: {e}")
            return
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def server_stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    # -- modes
    def enter_setup(self, why):
        log(f"[wifi] entering setup mode: {why}")
        self.mode = "setup"
        self.portal.offline_since_txt = time.strftime("%H:%M", time.localtime(
            time.time() - (self.clock() - self.offline_since)))
        self.scan()
        self.server_start()
        self.hotspot_up()
        self.last_retry = self.clock()

    def leave_setup(self, why):
        log(f"[wifi] leaving setup mode: {why}")
        self.hotspot_down()
        self.server_stop()
        with self.portal.lock:
            self.portal.message, self.portal.error, self.portal.pending = "", False, None
        self.mode = "normal"
        self.ever_online = True
        self.offline_since = None

    def try_connect(self, ssid, pw, hidden):
        log(f"[wifi] connecting to '{ssid}'")
        self.sleep(3)            # let the phone receive the "Connecting…" page first
        self.hotspot_down()
        name = f"wifi-{ssid}"[:60]
        self.nm("connection", "delete", name)
        args = ["device", "wifi", "connect", ssid]
        if pw:
            args += ["password", pw]
        args += ["ifname", IFACE, "name", name]
        if hidden:
            args += ["hidden", "yes"]
        code, out = self.nm(*args, timeout=60)
        if code == 0:
            # never give up retrying this network, and prefer it over older saved ones
            self.nm("connection", "modify", name, "connection.autoconnect-retries", "0",
                    "connection.autoconnect-priority", "10")
            if self.wait_online(CONNECT_WAIT_S):
                self.leave_setup(f"connected to '{ssid}'")
                return
            reason = (f"Joined '{ssid}' but there is no internet through it. It may need a login web page, "
                      "which the Pi can't do. The network is saved and will be retried.")
        else:
            self.nm("connection", "delete", name)   # wrong password etc.: don't keep a broken profile
            low = out.lower()
            if "secrets were required" in low or "password" in low or "802-1x" in low or "psk" in low:
                reason = f"Couldn't join '{ssid}': the password seems wrong (or it needs a username, which isn't supported)."
            elif "no network with ssid" in low:
                reason = f"Couldn't find '{ssid}'. Check the name and that it is in range, then refresh the list."
            else:
                reason = f"Couldn't join '{ssid}': {out[-160:]}"
        log(f"[wifi] {reason}")
        with self.portal.lock:
            self.portal.message, self.portal.error = reason, True
        self.scan()
        self.hotspot_up()
        self.last_retry = self.clock()

    def retry_saved(self):
        log("[wifi] nobody is using the setup page: retrying saved networks for 45 s")
        self.hotspot_down()
        self.nm("device", "connect", IFACE, timeout=40)   # NetworkManager picks the best saved network
        if self.wait_online(RETRY_FOR_S):
            self.leave_setup("a saved network is back")
            return
        self.scan()
        self.hotspot_up()
        self.last_retry = self.clock()

    def step(self):
        now = self.clock()
        if self.mode == "normal":
            forced = os.path.exists(TRIGGER)
            if forced:
                try:
                    os.remove(TRIGGER)
                except OSError:
                    pass
            if self.online():
                if self.offline_since is not None or not self.ever_online:
                    log("[wifi] online")
                self.ever_online = True
                self.offline_since = None
            elif self.offline_since is None:
                self.offline_since = now
                log("[wifi] offline: NetworkManager will keep retrying saved networks")
            if forced:
                self.offline_since = self.offline_since or now
                self.enter_setup("requested by hand")
                return
            if self.offline_since is not None:
                limit = OFFLINE_TO_SETUP_S if self.ever_online else BOOT_GRACE_S
                if now - self.offline_since >= limit:
                    self.enter_setup(f"offline for {int(now - self.offline_since)} s")
            return

        # setup mode
        with self.portal.lock:
            pending, self.portal.pending = self.portal.pending, None
        if pending:
            self.try_connect(*pending)
            return
        if self.portal.rescan_requested:
            self.portal.rescan_requested = False
            self.sleep(2)
            self.hotspot_down()
            self.scan()
            self.hotspot_up()
            return
        if self.online():           # e.g. Ethernet plugged in
            self.leave_setup("internet is back (Ethernet or another route)")
            return
        if now - self.last_retry >= RETRY_EVERY_S and now - self.portal.last_hit >= IDLE_BEFORE_RETRY_S:
            self.retry_saved()

    def run(self):
        log(f"[wifi] started; setup hotspot name '{self.hotspot_name}'")
        self.nm("connection", "down", HOTSPOT_CON)   # never start with a leftover hotspot
        try:
            while True:
                try:
                    self.step()
                except Exception as e:
                    log(f"[wifi] error: {e!r}")
                self.sleep(CHECK_EVERY_S if self.mode == "normal" else 5)
        finally:
            if self.mode == "setup":
                self.hotspot_down()
            self.server_stop()


if __name__ == "__main__":
    import signal
    import sys
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    WifiManager().run()
