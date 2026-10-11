"""Real HTTP authentication, CSRF checks, and cluster authorization."""
from http.cookiejar import CookieJar
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import build_opener, HTTPCookieProcessor, Request

from cluster_control.server import ControlServer
from cluster_control.service import APIError, Service
from cluster_control.store import Store


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.path = Path(self.root.name) / "control.sqlite3"
        self.store = Store(self.path)
        self.now = 1000
        self.service = Service(self.store, clock=lambda: self.now)
        self.owner, self.owner_token = self.register("Owner", "owner@example.com")
        self.peer, self.peer_token = self.register("Peer", "peer@example.com")
        self.cluster = self.service.create_cluster(self.owner["user"]["id"], {"name": "Lab"})["id"]

    def register(self, name, email):
        return self.service.register({"name": name, "email": email, "password": "a long test password"})

    def tearDown(self):
        self.store.close()
        self.root.cleanup()

    def test_authentication_expiry_and_logout(self):
        self.assertEqual(self.service.authenticate(self.owner_token)["user"], self.owner["user"])
        with self.assertRaisesRegex(APIError, "Invalid request token"):
            self.service.authenticate(self.owner_token, mutation=True)
        self.service.authenticate(self.owner_token, self.owner["csrf"], mutation=True)
        self.service.logout(self.peer_token)
        with self.assertRaises(APIError):
            self.service.authenticate(self.peer_token)
        self.now += self.service.SESSION_SECONDS
        with self.assertRaises(APIError):
            self.service.authenticate(self.owner_token)

    def test_passwords_and_session_tokens_are_not_stored_in_plaintext(self):
        with self.store.transaction() as db:
            self.assertNotEqual(db.execute("SELECT password FROM users LIMIT 1").fetchone()[0], "a long test password")
            self.assertNotIn(self.owner_token, [row[0] for row in db.execute("SELECT token_hash FROM sessions")])
        auth, _ = self.service.login({"email": "OWNER@EXAMPLE.COM", "password": "a long test password"})
        self.assertEqual(auth["user"]["id"], self.owner["user"]["id"])
        with self.assertRaisesRegex(APIError, "Invalid email or password"):
            self.service.login({"email": "owner@example.com", "password": "wrong"})

    def test_membership_approval_revocation_and_owner_protection(self):
        owner, peer = self.owner["user"]["id"], self.peer["user"]["id"]
        self.service.request_join(peer, self.cluster)
        self.assertEqual(self.service.clusters(peer)[0]["membership"], "pending")
        with self.assertRaisesRegex(APIError, "Approved cluster"):
            self.service.cluster(peer, self.cluster)
        with self.assertRaisesRegex(APIError, "owner access"):
            self.service.decide_membership(peer, self.cluster, peer, "approve")
        self.service.decide_membership(owner, self.cluster, peer, "approve")
        self.assertEqual(len(self.service.cluster(peer, self.cluster)["members"]), 2)
        with self.assertRaisesRegex(APIError, "owner's membership"):
            self.service.decide_membership(owner, self.cluster, owner, "revoke")
        self.service.decide_membership(owner, self.cluster, peer, "revoke")
        with self.assertRaises(APIError):
            self.service.cluster(peer, self.cluster)

    def test_persistence_and_duplicate_registration(self):
        with self.assertRaises(APIError) as error:
            self.register("Again", "OWNER@example.com")
        self.assertEqual(error.exception.status, 409)
        self.store.close()
        self.store = Store(self.path)
        self.service = Service(self.store, clock=lambda: self.now)
        self.assertEqual(self.service.authenticate(self.owner_token)["user"]["name"], "Owner")
        self.assertEqual(self.service.cluster(self.owner["user"]["id"], self.cluster)["name"], "Lab")

    def test_http_session_cookie_csrf_and_cross_origin(self):
        server = ControlServer(("127.0.0.1", 0), self.service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:%d" % server.server_port
        jar = CookieJar()
        client = build_opener(HTTPCookieProcessor(jar))

        def call(path, body=None, headers=None):
            request = Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                              headers={"Content-Type": "application/json", **(headers or {})})
            with client.open(request) as response:
                return response, json.load(response)

        try:
            with self.assertRaises(HTTPError) as error:
                call("/api/clusters")
            self.assertEqual(error.exception.code, 401)
            error.exception.close()
            response, auth = call("/api/auth/login", {"email": "owner@example.com", "password": "a long test password"})
            self.assertIn("HttpOnly", response.headers["Set-Cookie"])
            self.assertIn("SameSite=Strict", response.headers["Set-Cookie"])
            with self.assertRaises(HTTPError) as error:
                call("/api/clusters", {"name": "Unauthorized"})
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            with self.assertRaises(HTTPError) as error:
                call("/api/clusters", {"name": "Cross site"}, {"Origin": "https://evil.example", "X-CSRF-Token": auth["csrf"]})
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            _, created = call("/api/clusters", {"name": "HTTP lab"}, {"X-CSRF-Token": auth["csrf"]})
            self.assertIn("id", created)
            call("/api/auth/logout", {}, {"X-CSRF-Token": auth["csrf"]})
            with self.assertRaises(HTTPError) as error:
                call("/api/auth/me")
            error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
