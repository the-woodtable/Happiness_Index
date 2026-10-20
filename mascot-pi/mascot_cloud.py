#!/usr/bin/env python3
"""
Firebase uploader for the mascot meter. Standard library only.

Runs as a background thread inside mascot_meter.py. It signs in with the Pi's own Firebase account
(email/password, REST API, token refreshed automatically) so the database rules really apply to it.

What it writes (schema v1, see the Stage 1 contract):
  /live/status            every 5 s, server timestamp (drives "Live" / "Camera offline")
  /live/summary           when a number changes, at most every 2 s
  /live/timeline/points/i only the new or changed graph points
  /live/day, /live/dayStartMs   when the day changes
  /days/<day>/people/<key>      one per finished person, sent together with the matching summary
  /days/<day>/summary|timeline + /dayIndex/<day>   every 5 min for today, and finally at 04:00

Offline behaviour:
  * status / summary / timeline are never queued: the latest value replaces anything missed.
  * Person records are "queued" simply by not having been uploaded yet: they are already safe on disk
    in data/people/people_<day>.jsonl, and cloud_state.json remembers how far uploads got.
  * After 60 minutes offline (time the Pi was running without a connection), new people are marked
    q=false and are NOT queued. Totals stay complete because the summary is recomputed from the local
    file. Upload them later with:  python mascot_cloud.py backfill <YYYY-MM-DD>

Command line:
  python mascot_cloud.py backfill 2026-10-19     upload every person of that day (ignores the cap)
  python mascot_cloud.py state                   show upload progress
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

ENV_FILE = "/etc/mascot/firebase.env"
AUTH_URL = os.environ.get("MASCOT_AUTH_URL", "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword")
TOKEN_URL = os.environ.get("MASCOT_TOKEN_URL", "https://securetoken.googleapis.com/v1/token")

STATUS_EVERY_S = 5.0
SUMMARY_MIN_GAP_S = 2.0
DAY_COPY_EVERY_S = 300.0
RESCAN_EVERY_S = 60.0
QUEUE_CAP_S = 3600.0          # stop queuing people after this much time offline
PEOPLE_BATCH = 100
KEEP_DAYS = 30
CLOUD_PERSON_FIELDS = ("v", "firstSeenMs", "trackedS", "eyeContactS", "stayed", "reactionS",
                       "reactionChannel", "smiled", "timeToSmileS", "smileTimeS", "laughed",
                       "peakLaugh", "happiness", "shocked")


def log(msg):
    print(msg, flush=True)


class NetError(Exception):
    """No connection (DNS failure, timeout, refused). Distinct from an HTTP error reply."""


def http(method, url, body=None, form=None, timeout=10):
    headers = {}
    data = None
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode() or "null"
            return r.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise NetError(str(getattr(e, "reason", e)))


def read_env_file(path=ENV_FILE) -> dict:
    out = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def load_settings():
    file_vals = read_env_file()
    keys = ("FIREBASE_API_KEY", "FIREBASE_DB_URL", "FIREBASE_EMAIL", "FIREBASE_PASSWORD")
    vals = {k: os.environ.get(k) or file_vals.get(k, "") for k in keys}
    if not all(vals.values()) or any("PASTE" in v for v in vals.values()):
        return None
    vals["FIREBASE_DB_URL"] = vals["FIREBASE_DB_URL"].rstrip("/")
    return vals


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            try:
                out.append(json.loads(raw))
            except ValueError:
                out.append(None)   # keep line numbering; skipped on upload
    return out


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cloud_person(rec: dict) -> dict:
    out = {k: rec[k] for k in CLOUD_PERSON_FIELDS if k in rec}
    for k in ("trackedS", "eyeContactS", "reactionS", "timeToSmileS", "smileTimeS"):
        if k in out:
            out[k] = round(out[k], 1)
    for k in ("peakLaugh", "happiness"):
        if k in out:
            out[k] = round(out[k], 2)
    return out


class Firebase:
    """Signs in, keeps the ID token fresh, and talks to the Realtime Database REST API."""

    def __init__(self, s):
        self.key, self.db = s["FIREBASE_API_KEY"], s["FIREBASE_DB_URL"]
        self.email, self.password = s["FIREBASE_EMAIL"], s["FIREBASE_PASSWORD"]
        self.token = self.refresh_token = None
        self.expires = 0.0
        self.uid = None
        self.next_signin = 0.0

    def _sign_in(self):
        code, r = http("POST", f"{AUTH_URL}?key={self.key}",
                       {"email": self.email, "password": self.password, "returnSecureToken": True})
        if code != 200:
            msg = r.get("error", {}).get("message", r) if isinstance(r, dict) else r
            self.next_signin = time.monotonic() + 60
            raise RuntimeError(f"sign-in refused ({code}): {msg}. Check /etc/mascot/firebase.env")
        self.token, self.refresh_token = r["idToken"], r["refreshToken"]
        self.expires = time.monotonic() + int(r.get("expiresIn", 3600))
        if self.uid != r["localId"]:
            self.uid = r["localId"]
            log(f"[cloud] signed in as {self.email} (UID {self.uid})")

    def _refresh(self):
        code, r = http("POST", f"{TOKEN_URL}?key={self.key}",
                       form={"grant_type": "refresh_token", "refresh_token": self.refresh_token})
        if code != 200:
            self.token = self.refresh_token = None
            return self._sign_in()
        self.token, self.refresh_token = r["id_token"], r["refresh_token"]
        self.expires = time.monotonic() + int(r.get("expires_in", 3600))

    def ensure_token(self):
        if self.token and time.monotonic() < self.expires - 300:
            return
        if self.refresh_token:
            self._refresh()
        else:
            if time.monotonic() < self.next_signin:
                raise RuntimeError("waiting before retrying sign-in")
            self._sign_in()

    def request(self, method, path, body=None):
        """Returns (code, reply). Raises NetError when there is no connection."""
        self.ensure_token()
        url = f"{self.db}/{path}.json?auth={self.token}" if path else f"{self.db}/.json?auth={self.token}"
        code, r = http(method, url, body)
        if code == 401:
            self.expires = 0  # token expired early or was revoked: refresh and retry once
            self.ensure_token()
            url = f"{self.db}/{path}.json?auth={self.token}" if path else f"{self.db}/.json?auth={self.token}"
            code, r = http(method, url, body)
        return code, r

    def patch_root(self, payload: dict):
        return self.request("PATCH", "", payload)


class Cloud(threading.Thread):
    def __init__(self, settings, data_dir):
        super().__init__(daemon=True, name="cloud")
        self.fb = Firebase(settings)
        self.data_dir = data_dir
        self.people_dir = os.path.join(data_dir, "people")
        self.days_dir = os.path.join(data_dir, "days")
        self.state_path = os.path.join(data_dir, "cloud_state.json")
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.boot_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        # set by the meter
        self.day = None
        self.day_start = None
        self.people_count = {}
        self.summary = None
        self.rows = [[0, 0, 0, 0]]
        self.status = {"camera": "starting", "fps": 0.0, "faces": 0}
        self.rescan = threading.Event()

        # upload progress
        self.state = {"cursor": {}, "finalized": [], "offline_s": 0.0}
        try:
            with open(self.state_path) as f:
                self.state.update(json.load(f))
        except (OSError, ValueError):
            pass
        self.online = False
        self.ever_online = False
        self.last_err = ""
        self.sent_day = None
        self.sent_summary = None
        self.sent_rows = []
        self.last_summary_at = -1e9
        self.last_status_at = -1e9
        self.last_daycopy_at = -1e9
        self.last_daycopy = None
        self.last_rescan_at = -1e9
        self.last_state_save = time.monotonic()
        self.bad_batches = 0
        self.done_size = {}     # day -> people file size when we last reached its end
        self.pending_end = {}

    def _commit_cursor(self, day, new_cursor):
        if day in self.pending_end:
            self.done_size[day] = self.pending_end.pop(day)
        if new_cursor != self.state["cursor"].get(day, 0):
            self.state["cursor"][day] = new_cursor
            self.save_state()

    @classmethod
    def from_env(cls, data_dir):
        s = load_settings()
        return None if s is None else cls(s, data_dir)

    # ---- called from the meter thread ----
    def set_day(self, day, day_start, n_people):
        with self.lock:
            self.day, self.day_start = day, day_start
            self.people_count[day] = n_people

    def set_live(self, summary, rows):
        with self.lock:
            self.summary = summary
            self.rows = [list(r) for r in rows]

    def set_status(self, camera, fps, faces):
        with self.lock:
            self.status = {"camera": camera, "fps": round(float(fps), 1), "faces": int(faces)}

    def person_added(self, day, n_people):
        with self.lock:
            self.people_count[day] = n_people

    def day_finished(self, day):
        self.rescan.set()

    def rescan_days(self):
        self.rescan.set()

    def queue_open(self) -> bool:
        return self.state.get("offline_s", 0.0) < QUEUE_CAP_S

    def describe(self) -> str:
        with self.lock:
            waiting = sum(max(0, n - self.state["cursor"].get(d, 0)) for d, n in self.people_count.items())
        if self.online:
            return f"cloud online, {waiting} people waiting to upload"
        off = self.state.get("offline_s", 0.0) / 60
        extra = "" if self.queue_open() else " (queue CLOSED: new people stay local, use backfill)"
        why = f" [{self.last_err}]" if self.last_err else ""
        return f"cloud OFFLINE {off:.0f} min, {waiting} people waiting{extra}{why}"

    def stop(self):
        self.halt.set()
        self.join(timeout=8)
        self.save_state()

    # ---- state ----
    def save_state(self):
        try:
            cutoff = sorted(self.state["cursor"])[-KEEP_DAYS * 2:] if self.state["cursor"] else []
            self.state["cursor"] = {d: self.state["cursor"][d] for d in cutoff}
            self.state["finalized"] = sorted(set(self.state["finalized"]))[-KEEP_DAYS * 2:]
            atomic_write_json(self.state_path, self.state)
            self.last_state_save = time.monotonic()
        except OSError as e:
            log(f"[cloud] could not save state: {e!r}")

    def _went_offline(self, err):
        if self.online or err != self.last_err:
            log(f"[cloud] offline: {err}")
        self.online = False
        self.last_err = err

    def _went_online(self):
        if not self.online:
            log("[cloud] online" + ("" if self.ever_online else " (first upload since start)"))
            if self.ever_online and self.state.get("offline_s", 0) > 30:
                log(f"[cloud] back after {self.state['offline_s'] / 60:.1f} min offline")
        self.online = True
        self.ever_online = True
        self.last_err = ""
        if self.state.get("offline_s", 0) > 0:
            self.state["offline_s"] = 0.0
            self.save_state()

    # ---- main loop ----
    def run(self):
        last = time.monotonic()
        while not self.halt.is_set():
            t = time.monotonic()
            dt, last = t - last, t
            if not self.online:
                before = self.state["offline_s"]
                self.state["offline_s"] = before + dt
                if before < QUEUE_CAP_S <= self.state["offline_s"]:
                    log(f"[cloud] offline for {QUEUE_CAP_S / 60:.0f} min: no longer queuing new people "
                        "(they stay on the Pi; totals stay complete)")
                if t - self.last_state_save >= 60:
                    self.save_state()
            try:
                self.step(t)
            except NetError as e:
                self._went_offline(f"no connection ({e})")
            except RuntimeError as e:
                self._went_offline(str(e))
            except Exception as e:  # never let the uploader die
                self._went_offline(f"unexpected error {e!r}")
            self.halt.wait(0.5)
        # final best-effort push on shutdown
        try:
            if self.online:
                self.push_live(time.monotonic(), force=True)
        except Exception:
            pass

    def step(self, t):
        if t - self.last_status_at >= STATUS_EVERY_S:
            self.last_status_at = t
            with self.lock:
                st = dict(self.status)
            body = {"ts": {".sv": "timestamp"}, **st, "bootId": self.boot_id, "v": 1}
            code, r = self.fb.request("PUT", "live/status", body)
            if code != 200:
                raise RuntimeError(f"status write refused ({code}): {r}")
            self._went_online()
        if not self.online:
            return
        if self.day is None:
            return
        self.push_live(t)
        if t - self.last_daycopy_at >= DAY_COPY_EVERY_S:
            self.copy_today(t)
        if self.rescan.is_set() or t - self.last_rescan_at >= RESCAN_EVERY_S:
            self.rescan.clear()
            self.last_rescan_at = t
            self.finish_past_days()

    def _pending_people(self, day, count_hint=None, ignore_cap=False, limit=PEOPLE_BATCH):
        """(new_cursor, {path: record}) for the next batch of not-yet-uploaded people of a day."""
        cur = self.state["cursor"].get(day, 0)
        path = os.path.join(self.people_dir, f"people_{day}.jsonl")
        try:
            size = os.path.getsize(path)
        except OSError:
            return cur, {}
        if self.done_size.get(day) == size:   # nothing new since we last reached the end
            return cur, {}
        recs = read_jsonl(path)
        out = {}
        i = cur
        while i < len(recs) and len(out) < limit:
            r = recs[i]
            i += 1
            if not r or "key" not in r:
                continue
            if r.get("q", True) or ignore_cap:
                out[f"days/{day}/people/{r['key']}"] = cloud_person(r)
        if i >= len(recs):
            self.pending_end[day] = size   # becomes done_size once this batch is accepted
        return i, out

    def push_live(self, t, force=False):
        with self.lock:
            day, start = self.day, self.day_start
            summary = dict(self.summary) if self.summary else None
            rows = [list(r) for r in self.rows]
            count = self.people_count.get(day, 0)
        payload = {}
        new_day = self.sent_day != day
        if new_day:
            # fresh start or a new day: replace the whole live node
            payload["live/day"] = day
            payload["live/dayStartMs"] = start
            payload["live/timeline"] = {"points": rows}
        else:
            if len(rows) < len(self.sent_rows):
                payload["live/timeline"] = {"points": rows}
            else:
                for i, r in enumerate(rows):
                    if i >= len(self.sent_rows) or self.sent_rows[i] != r:
                        payload[f"live/timeline/points/{i}"] = r

        new_cursor, people = self._pending_people(day, count)
        summary_due = summary is not None and (
            new_day or people or force
            or (summary != self.sent_summary and t - self.last_summary_at >= SUMMARY_MIN_GAP_S))
        if not summary_due:
            # graph points only change when the counts change, so they travel with the summary
            payload = {k: v for k, v in payload.items() if not k.startswith("live/timeline")}
        if summary_due:
            payload["live/summary"] = {**summary, "updatedMs": int(time.time() * 1000)}
        payload.update(people)
        if not payload:
            self._commit_cursor(day, new_cursor)   # may have stepped over people that are not queued
            return
        code, r = self.fb.patch_root(payload)
        if code != 200:
            self.bad_batches += 1
            self.pending_end.pop(day, None)
            log(f"[cloud] live update refused ({code}): {str(r)[:200]}")
            if people and self.bad_batches >= 3:
                log(f"[cloud] skipping {len(people)} people of {day} that the database keeps refusing")
                self._commit_cursor(day, new_cursor)
                self.bad_batches = 0
            return
        self.bad_batches = 0
        self.sent_day = day
        if summary_due:
            self.sent_summary = summary
            self.last_summary_at = t
        if any(k.startswith("live/timeline") for k in payload):
            self.sent_rows = rows
        self._commit_cursor(day, new_cursor)
        if people:
            log(f"[cloud] uploaded {len(people)} people for {day}")

    def copy_today(self, t):
        """Keep /days/<today> and /dayIndex/<today> current, so history is complete even if 04:00 is missed."""
        with self.lock:
            day, summary, rows = self.day, self.summary, [list(r) for r in self.rows]
        if not summary:
            return
        snap = (day, json.dumps(summary, sort_keys=True), len(rows), rows[-1])
        if snap == self.last_daycopy:
            self.last_daycopy_at = t
            return
        idx = {k: summary.get(k, 0) for k in ("n", "stayed", "smiles", "laughs", "shocked")}
        code, r = self.fb.patch_root({
            f"days/{day}/summary": {**summary, "updatedMs": int(time.time() * 1000)},
            f"days/{day}/timeline": {"points": rows},
            f"dayIndex/{day}": idx,
        })
        if code == 200:
            self.last_daycopy, self.last_daycopy_at = snap, t
        else:
            log(f"[cloud] day copy refused ({code}): {str(r)[:200]}")
            self.last_daycopy_at = t

    def finish_past_days(self):
        """Upload remaining people of finished days, then their final summary/graph/index."""
        if not os.path.isdir(self.days_dir):
            return
        done = set(self.state["finalized"])
        for name in sorted(os.listdir(self.days_dir)):
            if not name.endswith(".json") or name.endswith(".tmp"):
                continue
            day = name[:-5]
            if day in done or day == self.day:
                continue
            try:
                age = (datetime.fromisoformat(self.day) - datetime.fromisoformat(day)).days
            except ValueError:
                continue
            if age > KEEP_DAYS:
                continue
            while True:  # people first, in batches
                before = self.state["cursor"].get(day, 0)
                new_cursor, people = self._pending_people(day)
                if people:
                    code, r = self.fb.patch_root(people)
                    if code != 200:
                        self.pending_end.pop(day, None)
                        log(f"[cloud] upload for {day} refused ({code}): {str(r)[:200]}")
                        return
                    log(f"[cloud] uploaded {len(people)} people for {day}")
                self._commit_cursor(day, new_cursor)
                if new_cursor == before:
                    break
            try:
                with open(os.path.join(self.days_dir, name)) as f:
                    final = json.load(f)
            except (OSError, ValueError):
                continue
            code, r = self.fb.patch_root({
                f"days/{day}/summary": final["summary"],
                f"days/{day}/timeline": final["timeline"],
                f"dayIndex/{day}": final["index"],
            })
            if code != 200:
                log(f"[cloud] final totals for {day} refused ({code}): {str(r)[:200]}")
                return
            self.state["finalized"].append(day)
            self.save_state()
            log(f"[cloud] {day} finalised in the cloud ({final['index'].get('n', 0)} people)")


# ---- command line ---------------------------------------------------------------
def cmd_backfill(args):
    s = load_settings()
    if s is None:
        sys.exit(f"No Firebase settings. Fill in {ENV_FILE} (or run with sudo if you cannot read it).")
    fb = Firebase(s)
    path = os.path.join(args.data_dir, "people", f"people_{args.day}.jsonl")
    recs = [r for r in read_jsonl(path) if r and "key" in r]
    if not recs:
        sys.exit(f"No people found in {path}")
    log(f"Uploading {len(recs)} people for {args.day} ...")
    for i in range(0, len(recs), PEOPLE_BATCH):
        chunk = {f"days/{args.day}/people/{r['key']}": cloud_person(r) for r in recs[i:i + PEOPLE_BATCH]}
        code, r = fb.patch_root(chunk)
        if code != 200:
            sys.exit(f"Refused ({code}): {r}")
        log(f"  {min(i + PEOPLE_BATCH, len(recs))}/{len(recs)}")
    final = os.path.join(args.data_dir, "days", f"{args.day}.json")
    if os.path.exists(final):
        with open(final) as f:
            fin = json.load(f)
        code, r = fb.patch_root({f"days/{args.day}/summary": fin["summary"],
                                 f"days/{args.day}/timeline": fin["timeline"],
                                 f"dayIndex/{args.day}": fin["index"]})
        log(f"Final totals: HTTP {code}")
    log("Done.")


def cmd_state(args):
    p = os.path.join(args.data_dir, "cloud_state.json")
    try:
        with open(p) as f:
            st = json.load(f)
    except (OSError, ValueError):
        sys.exit(f"No upload state yet at {p}")
    print(f"offline time counted: {st.get('offline_s', 0) / 60:.1f} min (queue closes at 60)")
    print("days finalised in the cloud:", ", ".join(st.get("finalized", [])) or "none")
    for day, cur in sorted(st.get("cursor", {}).items()):
        n = len(read_jsonl(os.path.join(args.data_dir, "people", f"people_{day}.jsonl")))
        print(f"  {day}: {cur}/{n} people processed for upload")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Mascot meter cloud tools")
    ap.add_argument("--data-dir", default=os.path.join(here, "data"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="upload every person of a day")
    b.add_argument("day", help="YYYY-MM-DD")
    sub.add_parser("state", help="show upload progress")
    args = ap.parse_args()
    {"backfill": cmd_backfill, "state": cmd_state}[args.cmd](args)


if __name__ == "__main__":
    main()
