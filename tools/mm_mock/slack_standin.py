"""A stand-in for the Slack Web API (the calls flakeid/scope/notify.py makes), for tests: it checks the bot token and
records every message, thread reply and picture.

    srv, calls = start(port, token="xoxb-test", users={"you@lab.edu": "U1"})
    os.environ["FLAKEID_SLACK_API"] = f"http://127.0.0.1:{port}/api/"
"""
from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def start(port, token="xoxb-test", users=None, names=None):
    users, names = users or {}, names or {}
    calls, state = [], {"ts": 1000, "files": {}}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _out(self, d, code=200):
            b = json.dumps(d).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            if self.path.startswith("/upload/"):
                with lock:
                    state["files"][self.path.split("/")[-1]] = len(raw)
                return self._out({"ok": True})
            if self.headers.get("Authorization") != "Bearer " + token:
                return self._out({"ok": False, "error": "invalid_auth"})
            m = self.path.split("/api/")[-1]
            ct = self.headers.get("Content-Type", "")
            d = dict(urllib.parse.parse_qsl(raw.decode())) if "urlencoded" in ct else json.loads(raw or b"{}")
            with lock:
                calls.append((m, d))
                if m == "users.lookupByEmail":
                    u = users.get(d.get("email", "").lower())
                    return self._out({"ok": True, "user": {"id": u}} if u else {"ok": False, "error": "users_not_found"})
                if m == "users.list":                  # (two pages, as Slack paginates)
                    mem = [{"id": uid, "name": em.split("@")[0], "profile": {"real_name": nm, "display_name": nm.split()[0]}}
                           for em, (uid, nm) in ((e, (v, names.get(v, e))) for e, v in users.items())]
                    half = len(mem) // 2
                    if d.get("cursor") == "p2":
                        return self._out({"ok": True, "members": mem[half:], "response_metadata": {"next_cursor": ""}})
                    return self._out({"ok": True, "members": mem[:half], "response_metadata": {"next_cursor": "p2"}})
                if m == "conversations.open":
                    return self._out({"ok": True, "channel": {"id": "D" + d["users"]}})
                if m == "chat.postMessage":
                    state["ts"] += 1
                    ch = d["channel"] if not d["channel"].startswith("#") else "C" + d["channel"][1:].upper()
                    return self._out({"ok": True, "channel": ch, "ts": f"{state['ts']}.0001"})
                if m == "files.getUploadURLExternal":
                    fid = f"F{len(state['files']) + len([c for c in calls if c[0] == m])}"
                    return self._out({"ok": True, "upload_url": f"http://127.0.0.1:{port}/upload/{fid}", "file_id": fid})
                if m == "files.completeUploadExternal":
                    return self._out({"ok": True, "files": d.get("files")})
            return self._out({"ok": False, "error": "unknown_method"})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, calls, state
