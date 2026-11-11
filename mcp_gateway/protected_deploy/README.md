# JANUS MCP / Protected Deployment Candidate — not deployed

Research-only, least-privilege Docker Compose deployment blueprint. It does **not** modify a NAS or public service by itself. Actual rollout requires an approved host, domain/TLS trust, operator-created credentials, backup/rollback and end-to-end acceptance. In prior NAS agreements, changes require specific user consent; no NAS container is touched in this PR.

## Network model

```text
Authorized user -- HTTPS TLS (operator cert) --> 127.0.0.1:8443 only
                                        Nginx Basic Auth + rate/connection limits
                                        read-only reverse proxy /mcp
                                                     |
                                                     v
                                      janus-mcp:8000 internal Docker DNS
                                        READ-ONLY / JANUS_ALLOW_WRITES=0
```

No external bind by default (`127.0.0.1:8443`), no Docker host network, and no direct gateway published port. The Nginx image version must be pinned by digest and audited **before production**, and TLS should use an operator-owned publicly trusted certificate when exposed beyond loopback. Do not expose Basic Auth over HTTP.

## What must be supplied locally (never commit to GitHub)

Place only in the trusted host's working directory, beneath `mcp_gateway/protected_deploy/secrets/`: `cert.pem`, `key.pem`, and `users.htpasswd` with a bcrypt password hash (e.g. generated with a reviewed password utility). Set their owner/permissions restrictively. These files are not part of the repository or PR artifacts. `compose.yml` deliberately fails to start without them; never commit actual secrets or a `.env` containing tokens.

Preflight before any deployment:

1. Confirm the intended host and owner permission. Inventory existing Docker services and backups. **Do not replace or restart NAS containers without explicit approval**.
2. Verify certificate chain/expiry and identity, password-hash strength, name resolution, firewall binding and Docker image digest. Configure approved login rate limiting and external client authorization at the host ingress.
3. Keep `JANUS_PUBLIC_EVIDENCE_ENABLED=0` until the client request approvals and privacy review pass; `JANUS_ALLOW_WRITES=0` must remain. Any `OPENALEX_API_KEY` is injected only through the trusted host secret environment and not PR CI.
4. Validate syntax with `docker compose -f mcp_gateway/protected_deploy/compose.yml config -q`. Only on an approved host, run `docker compose -f mcp_gateway/protected_deploy/compose.yml up --build -d`; this PR does not execute that command.
5. Test that unauthenticated `https://127.0.0.1:8443/mcp` returns 401, wrong credentials return 401, rate-limit floods return 429, and valid authenticated read-only MCP tools respond. Check access-log redaction. Verify no gateway is reachable directly on host port 8000.
6. Test per-query public-source approval, GDELT cooldown behavior, and failed-provider source status; inspect service health before enabling live source search. Roll back by stopping only this isolated stack if necessary.

## Important limitations

- The Nginx per-IP limiter is 12 requests/minute, burst 3, two simultaneous connections. It is a coarse local safeguard, not per-user budgeting or comprehensive DDoS protection; a trusted host ingress must impose total budgets and auditable identity.
- The GDELT DOC rate pacer in code enforces a 6-second gap *per process* and a 60-second backoff on HTTP 429. Multi-process deployments require a shared limiter (e.g. Redis-based or gateway-level); do not try to bypass provider quotas.
- Standalone GDELT rate limiting/HTTP 429 does not establish reliable online news retrieval. In prior GitHub Actions runs, GDELT returned HTTP 429. Use provider-sanctioned batch data for large-scale analysis.
- TLS, Nginx Basic Auth and limit_req protect the transport; they do **not** independently prove that the caller received the user's consent for public data egress. `public_query_approved=true` must be set by a trusted caller after checking consent.
- A retrieved source is not scientific evidence of a retrocausal signal. Live external source output must never be automatically fed into training without rights, provenance, independent validation and held-out separation.
- Actual source integration is a **draft** PR; Docker preflight tests are not proof of real server uptime or safe internet exposure.

Status: `PROTECTED_DEPLOYMENT_TEMPLATE / AUTH_TLS_GUARDS / NOT_DEPLOYED / HOST_APPROVAL_REQUIRED`.
