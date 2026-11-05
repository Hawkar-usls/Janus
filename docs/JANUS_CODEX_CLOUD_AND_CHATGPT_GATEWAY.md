# JANUS × Codex Cloud × Sign in with ChatGPT

**Status:** REVIEWABLE FOUNDATION ONLY. No ChatGPT account is connected by
committing this documentation. No Codex model call or GitHub code change is
authorized by a gateway handoff alone.

## Track A — Codex Cloud with your ChatGPT account

1. Visit [Codex](https://chatgpt.com/codex) and sign in to **your own**
   ChatGPT account; do not give JANUS a password, browser cookie, OAuth token
   or screenshot containing credentials.
2. Create and publish a Codex Cloud environment for
   `Hawkar-usls/Janus`. Authorize that exact GitHub repository in your own
   GitHub connection if the Codex environment asks.
3. Start a new Codex Cloud task with the authorized scope and read `AGENTS.md`.
   For an example, ask: *Review the gateway's offline tests, repair one
   reproducible failure restricted to mcp_gateway and mcp_gateway/tests,
   run those tests, and prepare a PR; do not merge or deploy.*
4. Review the PR and its CI before merging. Cloud work can continue even while
   your PC is off. A created environment does not itself initiate a task,
   configure background polling, or grant permission to other repos.

JANUS MCP now exposes a **read-only** tool:
`janus.codex_cloud_handoff(task, paths)`. It prepares a deterministic,
source-scoped prompt and ID for the owner to submit in Codex Cloud. It does
**not** launch Codex, spend usage, open a PR, or use GitHub write credentials.

## Track B — own open-source SIWC + Codex app-server bridge

This repository includes `mcp_gateway/codex_app_server.py` with a fixed
provider configuration and a fail-closed `turn/completed` checker. It does
not yet implement a browser OAuth callback or any live inference. Required
trusted-owner steps for the next reviewed code change:

1. Establish a persistent *host identity* for each real client/VM. This is
   distinct from the OAuth client ID. Do not generate a new host ID per
   worker, call, or restart.
2. Implement the documented **Sign in with ChatGPT** open-source dynamic
   registration: system browser, loopback on `127.0.0.1`, PKCE S256, fresh
   state and nonce, a newly issued OAuth `client_id`. No API key or client
   secret is needed for this eligible open-source flow.
3. Exchange the code and validate the signed ID token with OpenAI JWKS
   (issuer, audience, expiry, nonce), verify identity continuity, and check
   granted scopes including `chatgpt.tokens.use.direct`. Declined scope is
   an expected HOLD, not a reason to bypass consent.
4. Keep credentials and rotating refresh tokens in an owner-controlled,
   access-restricted credential store. Never commit, echo, log, place in
   workflow artifacts, or send into model prompts. Refresh and revoke
   sessions safely. Explicitly review the security of any remote token transfer.
5. After legitimate consent, the trusted client supplies the OAuth access
   token as `ACCESS_TOKEN` **only in the Codex app-server child process
   environment**; use `app_server_argv()` as the launch configuration.
   App-server communicates locally via stdio and sends Responses requests.
   Check `initialize`/`initialized`, `thread/start` and `turn/start`,
   then accept only a successful `turn/completed`. One completed model turn
   alone does not verify, deploy or safely merge a patch.
6. Put Codex execution in a disposable, tightly scoped workspace. Re-read
   exact Git commits and output diff, verify allowed paths and independent
   tests, then open a **review-only** PR. Do not give the model a general
   GitHub token or permission to change protected branches.

**Important host distinction:** A GitHub Actions runner is ephemeral. Do not
assume it can safely act as a persistent SIWC host or share a refresh token
with arbitrary untrusted jobs. This design requires further reviewed auth,
runtime, approval and token isolation before enabling automatic execution.
The default MCP server remains read-only and must not be exposed on the public
Internet without separate, tested transport authentication.

## Existing JANUS gateway boundaries

- `mcp_gateway/server.py` uses `JANUS_ALLOW_WRITES=0` by default.
- Existing write tools only target checkpoints on an expressly configured
  branch; they do **not** grant self-coding permission in other repositories.
- New `janus.codex_app_server_readiness()` reports
  `AUTHENTICATION_NOT_CONNECTED` until a separate verified auth integration
  exists. Never treat catalog or unverified account claims as entitlement.
- This phase is architecture and task preparation, not `JANUS_AUTONOMY_FULL`.
  No models, credits, tokens, new recurring jobs, NAS or external organs are
  automatically affected.

## References

- Codex Cloud: https://help.openai.com/en/articles/20001545-using-codex-cloud
- Open-source Sign in with ChatGPT: https://developers.openai.com/siwc/token-sharing-open-source
- Dynamic registration / PKCE: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
- Official Codex app-server settings: https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server
- Preview limitations: https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
