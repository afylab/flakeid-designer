"""Messages to you when the scan needs you or a job ends: Slack (a bot: a direct message to the person running the scan,
one thread per run, with pictures; or an incoming webhook) and / or email (SMTP).

Slack bot (the app "FlakeID Scanner": scopes chat:write, im:write, files:write, users:read, users:read.email):
  notify_slack_token_file  a file holding its bot token (xoxb-...), readable only by the scan's Windows account; or the
                           environment variable FLAKEID_SLACK_TOKEN (never the settings file, which is copied around)
  notify_to                who runs the scan: their Slack email, name (display or full, if only one member has it) or
                           member ID (the start screen's "Messages to", remembered in <local_root>/notify.json); found
                           in Slack once (users.lookupByEmail / users.list), cached in <local_root>/slack_users.json
  notify_slack_channel     optional: a channel ("#microscope" or its ID) that gets the run's thread too (invite the bot)
  Only what needs you, each as a new message (so each one notifies): an objective change / the IR setup / a focus
  question (with the flake's 10x picture) / the stacking-domain question, "scanning finished" (the microscope is free),
  "export finished", stopped / failed.  (notify_threads: a run's messages in one thread instead; notify_on "progress":
  each chip's plan too.)
Webhook / email:
  notify_slack_webhook   "https://hooks.slack.com/services/..." (a Slack app's Incoming Webhooks: one channel or DM)
  notify_email_to        "you@physics.ucsb.edu" (several: comma-separated)
  notify_email_from      the sender address (default: notify_smtp_user)
  notify_smtp_host       e.g. "smtp.gmail.com"; port notify_smtp_port (587: STARTTLS, 465: SSL, 25: plain)
  notify_smtp_user       login, if the server needs one; the password comes from the environment variable
                         FLAKEID_SMTP_PASSWORD or the file notify_smtp_password_file (not the settings file, which is
                         copied around); for Gmail an app password
  notify_on              which events: "operator" (an objective change, the IR setup, a focus question), "finished",
                         "stopped", "failed"

Sending runs in the background (a slow or unreachable server never holds up the scan); failures are logged.  The
same message is sent at most once a minute.

    python -m flakeid.scope.notify --settings C:\\FlakeID\\scan_settings.json --test [--to "Your Name"]
"""
from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

_last = {}
_lock = threading.Lock()


def email_ready(s):
    """Email can be sent from this PC (an SMTP server in the settings)."""
    return bool(getattr(s, "notify_smtp_host", ""))


def configured(s):
    return bool(getattr(s, "notify_slack_webhook", "") or (getattr(s, "notify_email_to", "") and getattr(s, "notify_smtp_host", ""))
                or bot_ready(s))


# ---------------------------------------------------------------------------------------------------- Slack bot
SLACK_API = os.environ.get("FLAKEID_SLACK_API", "https://slack.com/api/")    # (a stand-in server for tests)


def slack_token(s):
    t = os.environ.get("FLAKEID_SLACK_TOKEN", "").strip()
    f = getattr(s, "notify_slack_token_file", "")
    if not t and f:
        try:
            with open(f, encoding="utf-8") as fh:
                t = fh.read().strip()
        except OSError:
            t = ""
    return t


def bot_ready(s):
    return bool(slack_token(s) and (getattr(s, "notify_to", "") or getattr(s, "notify_slack_channel", "")))


class SlackBot:
    """The few Slack Web API calls the scan needs (urllib only: Python 3.8, Windows 7)."""

    def __init__(self, token, cache=None):
        self.token, self.cache = token, cache

    def call(self, method, data=None, form=False):
        url = SLACK_API.rstrip("/") + "/" + method
        h = {"Authorization": "Bearer " + self.token}
        if form:
            body = urllib.parse.urlencode(data or {}).encode()
            h["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = json.dumps(data or {}).encode("utf-8")
            h["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(url, data=body, headers=h)
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as r:
            out = json.loads(r.read().decode("utf-8") or "{}")
        if not out.get("ok"):
            raise RuntimeError(f"Slack {method}: {out.get('error', 'not ok')}")
        return out

    def list_members(self):
        """The workspace's people (not bots, not deactivated): Slack's member records (users.list: scope users:read)."""
        members, cursor = [], ""
        for _ in range(50):
            r = self.call("users.list", dict({"limit": 200}, **({"cursor": cursor} if cursor else {})), form=True)
            members += [m for m in r.get("members") or [] if not m.get("deleted") and not m.get("is_bot") and m.get("id") != "USLACKBOT"]
            cursor = (r.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                break
        return members

    def find_by_name(self, name):
        """The member whose display name or full name is name (case does not matter; else the only one it starts);
        a name that fits several people is refused with who they are."""
        q, members = " ".join(name.lower().split()), self.list_members()

        def names(m):
            p = m.get("profile") or {}
            return {" ".join(str(v).lower().split()) for v in (p.get("display_name"), p.get("real_name"), m.get("real_name"), m.get("name")) if v}
        hit = [m for m in members if q in names(m)] or [m for m in members if any(v.startswith(q) for v in names(m))]
        if not hit:
            raise RuntimeError(f"no Slack member called {name!r} (use their Slack email instead)")
        if len(hit) > 1:
            who = ", ".join(f"{(m.get('profile') or {}).get('real_name') or m.get('name')}" for m in hit[:6])
            raise RuntimeError(f"{name!r} fits {len(hit)} Slack members ({who}): give the full name or their email")
        return hit[0]["id"]

    def user_id(self, who):
        """A Slack member ID from an email, a name or an ID (U...); cached in <local_root>/slack_users.json."""
        who = who.strip()
        key = who.lower()
        C = {}
        if self.cache:
            try:
                C = json.loads(Path(self.cache).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                C = {}
            if C.get(key):
                return C[key]
        if "@" in who:
            uid = self.call("users.lookupByEmail", {"email": key}, form=True)["user"]["id"]
        elif who[:1] in "UW" and who[1:].isalnum() and who.upper() == who and len(who) >= 9:
            uid = who
        else:
            uid = self.find_by_name(who)
        if self.cache:
            try:
                C[key] = uid
                Path(self.cache).write_text(json.dumps(C, indent=1), encoding="utf-8")
            except OSError:
                pass
        return uid

    def dm(self, who):
        return self.call("conversations.open", {"users": self.user_id(who)})["channel"]["id"]

    def post(self, channel, text, thread_ts=None, broadcast=False):
        d = {"channel": channel, "text": text, "unfurl_links": False}
        if thread_ts:
            d["thread_ts"] = thread_ts
            if broadcast:
                d["reply_broadcast"] = True
        r = self.call("chat.postMessage", d)
        return r["channel"], r["ts"]

    def upload(self, channel, path, title, comment, thread_ts=None):
        """A picture into the conversation (thread), with comment: the files.getUploadURLExternal way."""
        data = Path(path).read_bytes()
        u = self.call("files.getUploadURLExternal", {"filename": Path(path).name, "length": len(data)}, form=True)
        req = urllib.request.Request(u["upload_url"], data=data, headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
            if r.status >= 300:
                raise RuntimeError(f"Slack upload: HTTP {r.status}")
        d = {"files": [{"id": u["file_id"], "title": title}], "channel_id": channel, "initial_comment": comment}
        if thread_ts:
            d["thread_ts"] = thread_ts
        self.call("files.completeUploadExternal", d)


def members(s, max_age_h=24.0):
    """The workspace's people for the start screen's name completion: [{id, name, handle}], from
    <local_root>/slack_members.json if fetched within max_age_h, else from Slack (then saved); [] without the bot."""
    f = Path(getattr(s, "local_root", "") or ".") / "slack_members.json"
    old = None
    try:
        old = json.loads(f.read_text(encoding="utf-8"))
        if time.time() - float(old.get("t", 0)) < max_age_h * 3600:
            return old["members"]
    except (OSError, ValueError, KeyError):
        pass
    if not slack_token(s):
        return (old or {}).get("members", [])
    try:
        out = []
        for m in SlackBot(slack_token(s)).list_members():
            p = m.get("profile") or {}
            name = p.get("real_name") or m.get("real_name") or p.get("display_name") or m.get("name") or ""
            out.append(dict(id=m["id"], name=name, handle=p.get("display_name") or m.get("name") or ""))
        out.sort(key=lambda d: d["name"].lower())
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(dict(t=time.time(), members=out)), encoding="utf-8")
        except OSError:
            pass
        return out
    except Exception:
        return (old or {}).get("members", [])               # (no network: the last list)


_q = queue.Queue()
_run = {"current": None}          # the latest run's handle ({"dests": [(channel id, ts)]}); a run that finishes in the
                                  # background (its export) keeps its own handle, so its last messages go to its thread


def _worker():
    while True:
        fn, log = _q.get()
        try:
            fn()
        except Exception as e:
            if log:
                log(f"  (Slack message not sent: {type(e).__name__}: {e})")
        finally:
            _q.task_done()


threading.Thread(target=_worker, daemon=True).start()


def _bot(s):
    cache = Path(getattr(s, "local_root", "") or ".") / "slack_users.json"
    return SlackBot(slack_token(s), cache=str(cache))


def _dest_list(s, bot):
    out = []
    if getattr(s, "notify_to", ""):
        out.append(bot.dm(s.notify_to))
    if getattr(s, "notify_slack_channel", ""):
        out.append(s.notify_slack_channel)
    return out


def start_run(s, title, text, log=None):
    """A run starts: its thread in the DM (and the channel); what is sent for it (send(..., run=handle), or anything
    without a run until the next start_run) is a reply in it.  Returns the run's handle (None without the bot)."""
    if not bot_ready(s) or not getattr(s, "notify_threads", False):
        return None                                     # (no threads: every message a new one, so each notifies)
    host = socket.gethostname()
    H = {"dests": None}
    _run["current"] = H

    def go():
        bot = _bot(s)
        dests = []
        for ch in _dest_list(s, bot):
            try:
                dests.append(bot.post(ch, f"*{title}*\n{text}\n_{time.strftime('%Y-%m-%d %H:%M')}, {host}_"))
            except Exception as e:
                if log:
                    log(f"  (Slack: {ch} not reached: {e})")
        H["dests"] = dests
    _q.put((go, log))
    return H


def end_run(run=None):
    """The run's thread is closed: messages without a run go out as new ones again."""
    if run is None or _run["current"] is run:
        _run["current"] = None


def bot_send(s, kind, subject, text, image=None, log=None, broadcast=False, run=None):
    """A message (and a picture) to the run's thread (run: its handle; default the latest run), or a new message."""
    if not bot_ready(s):
        return
    H = run if run is not None else _run["current"]

    def go():
        bot = _bot(s)
        dests = H["dests"] if H is not None else None
        if dests is None:
            dests = [(ch, None) for ch in _dest_list(s, bot)]
        msg = f"*{subject}*\n{text}"
        for ch, ts in dests:
            if image and Path(image).is_file():
                bot.upload(ch, image, Path(image).name, msg, thread_ts=ts)
            else:
                bot.post(ch, msg, thread_ts=ts, broadcast=broadcast and ts is not None)
        if log and kind == "test":
            log(f"  Slack bot message sent to {len(dests)} conversation(s)")
    _q.put((go, log))


def job_end(sc, ok, stopped, mins):
    """The end of a job (the scan program, after each one): finished / stopped / failed, where the results are; the run's
    thread closed.  Jobs shorter than notify_min_job_min send nothing (a flat field, a quick test)."""
    s = sc.s
    run = getattr(sc, "slack_run", None)
    if ok and getattr(sc, "export_bg", False):          # (its messages come from the run: scanning finished, then the export)
        end_run(run)
        return
    if mins >= s.notify_min_job_min:
        kind = "finished" if ok else ("stopped" if stopped else "failed")
        where = getattr(sc, "results_dest", None) or getattr(sc, "run_dir", None)
        send(s, kind, f"{getattr(sc, 'run_label', None) or s.name or s.mode}: {kind}",
             f"The {s.mode} job {kind} after {mins:.0f} min." + (f"  Results: {where}" if where else "")
             + ("" if ok else "  See the log in the scan program."), log=sc.say, run=run)
    end_run(run)


def flush(timeout=60):
    """Wait until the queued Slack messages are sent (the end of the program)."""
    t_end = time.time() + timeout
    while _q.unfinished_tasks and time.time() < t_end:
        time.sleep(0.1)


def _ssl_context():
    import ssl
    try:
        import certifi                                  # (Windows 7's own certificate store can be out of date)
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _slack(url, subject, text):
    body = json.dumps({"text": f"*{subject}*\n{text}"}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20, context=_ssl_context()) as r:
        if r.status >= 300:
            raise RuntimeError(f"Slack answered {r.status}")


def _password(s):
    pw = os.environ.get("FLAKEID_SMTP_PASSWORD", "")
    f = getattr(s, "notify_smtp_password_file", "")
    if not pw and f:
        with open(f, encoding="utf-8") as fh:
            pw = fh.read().strip()
    return pw


def _email(s, subject, text):
    import smtplib
    from email.message import EmailMessage
    to = [a.strip() for a in s.notify_email_to.split(",") if a.strip()]
    m = EmailMessage()
    m["Subject"] = subject
    m["From"] = s.notify_email_from or s.notify_smtp_user or to[0]
    m["To"] = ", ".join(to)
    m.set_content(text)
    port = int(s.notify_smtp_port or 587)
    if port == 465:
        srv = smtplib.SMTP_SSL(s.notify_smtp_host, port, timeout=20, context=_ssl_context())
    else:
        srv = smtplib.SMTP(s.notify_smtp_host, port, timeout=20)
    try:
        if port == 587:
            srv.starttls(context=_ssl_context())
        if s.notify_smtp_user:
            srv.login(s.notify_smtp_user, _password(s))
        srv.send_message(m)
    finally:
        try:
            srv.quit()
        except Exception:
            pass


def send(s, kind, subject, text, log=None, wait=False, image=None, run=None):
    """Send a message of kind (one of notify_on) by every configured way; in the background unless wait.  Returns the
    thread (or None when nothing is sent).  image: a picture for the Slack bot (the others get the text).  Kind
    "progress" (a chip's plan) goes to the Slack bot's thread only."""
    if not configured(s) or kind not in (getattr(s, "notify_on", None) or ()):
        return None
    if bot_ready(s):
        bot_send(s, kind, subject, text, image=image, log=log, broadcast=kind in ("scanned", "finished", "stopped", "failed"), run=run)
        if wait:
            flush()
    if kind == "progress":
        return None
    with _lock:
        now = time.time()
        key = (kind, text[:120])                        # (the same request repeated; different ones all go out)
        if kind != "test" and now - _last.get(key, 0.0) < 60.0:
            return None
        _last[key] = now
    host = socket.gethostname()
    subject = f"[{host}] {subject}"
    text = f"{text}\n\n({time.strftime('%Y-%m-%d %H:%M')}, {host})"

    def work():
        for name, fn in (("Slack", (lambda: _slack(s.notify_slack_webhook, subject, text)) if s.notify_slack_webhook else None),
                         ("email", (lambda: _email(s, subject, text)) if s.notify_email_to and s.notify_smtp_host else None)):
            if fn is None:
                continue
            try:
                fn()
                if log and kind == "test":
                    log(f"  {name} message sent")
            except Exception as e:
                if log:
                    log(f"  ({name} message not sent: {type(e).__name__}: {e})")
    t = threading.Thread(target=work, daemon=True)
    t.start()
    if wait:
        t.join(60)
    return t


def main(argv=None):
    import argparse
    from .settings import load
    ap = argparse.ArgumentParser(prog="python -m flakeid.scope.notify", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--settings", help="the scan settings file")
    ap.add_argument("--test", action="store_true", help="send a test message by every configured way")
    ap.add_argument("--to", help="Slack name or email to send it to (default: notify_to, or the one the start screen remembered)")
    a = ap.parse_args(argv)
    s = load(a.settings)
    if a.to:
        s.notify_to = a.to
    elif not s.notify_to:
        try:
            s.notify_to = json.loads((Path(s.local_root) / "notify.json").read_text(encoding="utf-8")).get("to", "")
        except (OSError, ValueError):
            pass
    if not configured(s):
        print("nothing configured: notify_slack_token_file + a name (--to), notify_slack_webhook, or notify_email_to + notify_smtp_host")
        return 1
    s.notify_on = tuple(s.notify_on or ()) + ("test",)
    send(s, "test", "FlakeIdentifier: test message", "If you can read this, the scan program can reach you.", log=print, wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
