"""Same-origin HTTP API and static dashboard; no third-party server dependencies."""
import argparse
from collections import defaultdict, deque
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

from cluster_control.service import APIError, Service
from cluster_control.store import Store


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service, secure_cookie=False, public_origin=None):
        self.service = service
        self.secure_cookie = secure_cookie
        self.public_origin = public_origin
        self.auth_attempts = defaultdict(deque)
        self.attempt_lock = threading.Lock()
        super().__init__(address, Handler)

    def service_actions(self):
        self.service.reconcile()


class Handler(BaseHTTPRequestHandler):
    server_version = "DTFM-Control/1"

    def log_message(self, format, *args):
        # Paths may contain identities; do not log cookies, request bodies or tokens.
        pass

    def do_GET(self):
        self.handle_request()

    def do_HEAD(self):
        self.command = "GET"
        self.head_only = True
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def do_PUT(self):
        self.handle_request()

    def cookie(self, token, clear=False):
        age = 0 if clear else self.server.service.SESSION_SECONDS
        return "dtfm_session=%s; Path=/; HttpOnly; SameSite=Strict; Max-Age=%d%s" % (
            token, age, "; Secure" if self.server.secure_cookie else "")

    def respond(self, status, body, cookie=None):
        payload = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        if not getattr(self, "head_only", False):
            self.wfile.write(payload)

    def read_body(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "Invalid content length") from None
        if not 0 < size <= 262144:
            raise APIError(413, "Request body must be between 1 byte and 256 KiB")
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            raise APIError(415, "Use application/json")
        try:
            body = json.loads(self.rfile.read(size), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeDecodeError):
            raise APIError(400, "Invalid JSON") from None
        if not isinstance(body, dict):
            raise APIError(400, "Expected a JSON object")
        return body

    def check_origin(self):
        origin = self.headers.get("Origin")
        expected = self.server.public_origin or (
            ("https" if self.server.secure_cookie else "http") + "://" + self.headers.get("Host", ""))
        if (origin and origin != expected) or self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise APIError(403, "Cross-origin requests are not allowed")

    def throttle_auth(self):
        with self.server.attempt_lock:
            now = time.monotonic()
            attempts = self.server.auth_attempts[self.client_address[0]]
            while attempts and attempts[0] < now - 60:
                attempts.popleft()
            if len(attempts) >= 20:
                raise APIError(429, "Too many sign-in attempts; try again in a minute")
            attempts.append(now)

    def handle_request(self):
        try:
            path = urlsplit(self.path).path
            mutation = self.command != "GET"
            if mutation:
                self.check_origin()
            body = self.read_body() if mutation else {}
            service = self.server.service
            if path == "/api/health" and not mutation:
                return self.respond(200, {"status": "ok"})
            if path in ("/api/auth/register", "/api/auth/login") and self.command == "POST":
                self.throttle_auth()
                result, token = (service.register(body) if path.endswith("register") else service.login(body))
                return self.respond(200, result, self.cookie(token))
            if path == "/api/worker/pair" and self.command == "POST":
                self.throttle_auth()
                return self.respond(200, service.pair_agent(body))
            if path == "/api/worker/heartbeat" and self.command == "POST":
                authorization = self.headers.get("Authorization", "")
                token = authorization[7:] if authorization.startswith("Bearer ") else None
                return self.respond(200, service.heartbeat(token, body.get("capabilities")))
            if path in ("/api/worker/poll", "/api/worker/report") and self.command == "POST":
                authorization = self.headers.get("Authorization", "")
                token = authorization[7:] if authorization.startswith("Bearer ") else None
                return self.respond(200, service.poll_worker(token) if path.endswith("poll") else service.report_worker(token, body))
            if not path.startswith("/api/"):
                return self.static(path)
            cookies = SimpleCookie()
            cookies.load(self.headers.get("Cookie", ""))
            token = cookies["dtfm_session"].value if "dtfm_session" in cookies else None
            auth = service.authenticate(token, self.headers.get("X-CSRF-Token"), mutation)
            user_id = auth["user"]["id"]
            if path == "/api/auth/me" and self.command == "GET":
                return self.respond(200, auth)
            if path == "/api/catalog" and self.command == "GET":
                from cluster_control.spec import DEFAULTS, MODELS
                return self.respond(200, {"defaults": DEFAULTS, "models": MODELS})
            if path == "/api/auth/logout" and self.command == "POST":
                service.logout(token)
                return self.respond(200, {"ok": True}, self.cookie("", clear=True))
            parts = path.strip("/").split("/")
            result = self.route_user(service, user_id, parts, body)
            self.respond(200, result)
        except APIError as error:
            self.respond(error.status, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            import traceback
            traceback.print_exc()
            self.respond(500, {"error": "Internal server error"})

    def route_user(self, service, user_id, parts, body):
        if parts == ["api", "clusters"]:
            if self.command == "GET":
                return {"clusters": service.clusters(user_id)}
            if self.command == "POST":
                return service.create_cluster(user_id, body)
        if len(parts) >= 3 and parts[:2] == ["api", "clusters"]:
            cluster_id = parts[2]
            if len(parts) == 3 and self.command == "GET":
                return service.cluster(user_id, cluster_id)
            if parts[3:] == ["join"] and self.command == "POST":
                return service.request_join(user_id, cluster_id)
            if len(parts) == 5 and parts[3] == "members" and self.command == "POST":
                return service.decide_membership(user_id, cluster_id, parts[4], body.get("decision"))
            if parts[3:] == ["pairing"] and self.command == "POST":
                return service.create_pairing(user_id, cluster_id)
            if parts[3:] == ["agents"] and self.command == "GET":
                return {"agents": service.agents(user_id, cluster_id)}
            if len(parts) == 6 and parts[3] == "agents" and parts[5] == "sharing" and self.command == "POST":
                return service.set_sharing(user_id, cluster_id, parts[4], body)
            if parts[3:] == ["preview"] and self.command == "POST":
                return service.preview_job(user_id, cluster_id, body)
            if parts[3:] == ["jobs"]:
                if self.command == "GET":
                    return {"jobs": service.jobs(user_id, cluster_id)}
                if self.command == "POST":
                    return service.create_job(user_id, cluster_id, body)
        if len(parts) == 3 and parts[:2] == ["api", "jobs"] and self.command == "GET":
            try:
                after = int(parse_qs(urlsplit(self.path).query).get("after", ["0"])[0])
            except ValueError:
                raise APIError(400, "Invalid event cursor") from None
            return service.job(user_id, parts[2], after)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and self.command == "POST":
            if parts[3] == "start":
                return service.start_job(user_id, parts[2])
            if parts[3] == "cancel":
                return service.cancel_job(user_id, parts[2])
        raise APIError(404, "Endpoint not found")

    def static(self, path):
        if self.command != "GET":
            raise APIError(405, "Static assets only support GET")
        files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                 "/app.js": ("app.js", "text/javascript"), "/styles.css": ("styles.css", "text/css"),
                 "/favicon.svg": ("favicon.svg", "image/svg+xml")}
        if path not in files:
            raise APIError(404, "Page not found")
        name, content_type = files[path]
        payload = (Path(__file__).parent / "web" / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if not getattr(self, "head_only", False):
            self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--database", default=".cluster-data/control.sqlite3")
    parser.add_argument("--secure-cookie", action="store_true", help="enable behind an HTTPS reverse proxy")
    parser.add_argument("--public-origin", help="exact browser origin when using a reverse proxy")
    args = parser.parse_args()
    store = Store(args.database)
    server = ControlServer((args.host, args.port), Service(store), args.secure_cookie, args.public_origin)
    print("DT-FM cluster dashboard: http://%s:%d" % server.server_address, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
