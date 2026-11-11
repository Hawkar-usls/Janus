"""Protected gateway deployment static fail-closed checks (never starts Docker)."""
from pathlib import Path
import unittest

ROOT=Path("mcp_gateway/protected_deploy")
COMPOSE=ROOT/"compose.yml"
CONF=ROOT/"nginx.conf"

class ProtectedMCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose=COMPOSE.read_text(encoding="utf8")
        cls.nginx=CONF.read_text(encoding="utf8")

    def test_tls_only_loopback_exposure(self):
        self.assertIn('"127.0.0.1:8443:8443"',self.compose)
        self.assertIn("listen 8443 ssl;",self.nginx)
        self.assertIn("ssl_protocols TLSv1.2 TLSv1.3;",self.nginx)
        self.assertNotIn("0.0.0.0:8443",self.compose)
        self.assertNotIn("80:80",self.compose)

    def test_gateway_private_and_write_disabled(self):
        self.assertIn('JANUS_ALLOW_WRITES: "0"',self.compose)
        self.assertIn('JANUS_PUBLIC_EVIDENCE_ENABLED: "${JANUS_PUBLIC_EVIDENCE_ENABLED:-0}"',self.compose)
        self.assertIn('expose:\n      - "8000"',self.compose)
        self.assertNotIn('"8000:8000"',self.compose)
        self.assertNotIn("network_mode: host",self.compose)
        self.assertIn("read_only: true",self.compose)
        self.assertIn("no-new-privileges:true",self.compose)

    def test_proxy_restricts_path_and_removes_auth_to_backend(self):
        self.assertIn("location = /mcp {",self.nginx)
        self.assertIn("location / {\n            return 404;",self.nginx)
        self.assertIn("proxy_pass http://janus-mcp:8000/mcp;",self.nginx)
        self.assertIn('proxy_set_header Authorization "";',self.nginx)
        self.assertIn("proxy_buffering off;",self.nginx)

    def test_ingress_has_auth_rate_and_concurrency_controls(self):
        for requirement in (
            'auth_basic "JANUS authorized MCP access";',
            'auth_basic_user_file /etc/nginx/auth/users.htpasswd;',
            'limit_req_zone $binary_remote_addr zone=janus_req:10m rate=12r/m;',
            'limit_conn_zone $binary_remote_addr zone=janus_conn:10m;',
            'limit_req zone=janus_req burst=3 nodelay;',
            'limit_conn janus_conn 2;',
            'client_max_body_size 64k;',
        ):
            self.assertIn(requirement,self.nginx)

    def test_real_secrets_are_not_committed(self):
        for name in ("cert.pem","key.pem","users.htpasswd"):
            self.assertFalse((ROOT/"secrets"/name).exists(),
                             "Do not commit TLS credentials or password hashes")
            self.assertIn("./secrets/"+name+":",self.compose)
        self.assertFalse((ROOT/".env").exists())

    def test_ci_never_deploys_or_touches_nas(self):
        workflow=Path(".github/workflows/public-evidence-gateway.yml").read_text()
        self.assertNotIn("docker compose up",workflow)
        self.assertNotIn("ssh ",workflow)
        self.assertNotIn("JANUS_ALLOW_WRITES=1",workflow)

if __name__=="__main__":
    unittest.main()
