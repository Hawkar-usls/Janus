# JANUS × Codex Cloud × Sign in with ChatGPT

**Status:** LOCAL CLIENT IMPLEMENTED; LIVE CONNECTION REQUIRES OWNER CONSENT.
Committing this code does not connect an account or verify model access.
Offline tests and CI do not establish a successful live model request.
No GitHub code change is authorized by a gateway handoff alone.

These are independent paths: a Cloud environment is required for Track A,
but is not a prerequisite for the local client in Track B.

## Track A — Codex Cloud with your ChatGPT account

1. Visit [Codex](https://chatgpt.com/codex) and sign in to **your own**
   ChatGPT account; do not give JANUS a password, browser cookie, OAuth token
   or screenshot containing credentials.
2. Choose **Work in → Cloud → Select environment → Create environment**.
   Select `Hawkar-usls/Janus`, complete GitHub authorization if prompted,
   let Codex prepare the setup and review its report. Resolve unfinished
   setup, save changes and select **Publish**. Then select **Start a new task**.
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

The separate human-operated `python -m mcp_gateway.codex_bridge` client now
implements browser OAuth, protected per-account credentials, renewal and
revocation, and isolated text inference through Codex app-server. It is
deliberately **not an MCP tool**. Remote callers cannot retrieve tokens,
start login or launch this runtime through the existing MCP read surface.
Its authentication/runtime boundary follows these requirements:

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
with arbitrary untrusted jobs. Automatic code execution still requires a
separately reviewed implementation and source-scope enforcement. This client
only enables bounded text inference.
The default MCP server remains read-only and must not be exposed on the public
Internet without separate, tested transport authentication.

## Existing JANUS gateway boundaries

- `mcp_gateway/server.py` uses `JANUS_ALLOW_WRITES=0` by default.
- Existing write tools only target checkpoints on an expressly configured
  branch; they do **not** grant self-coding permission in other repositories.
- New `janus.codex_app_server_readiness()` reports
  `AUTHENTICATION_NOT_CONNECTED` as a static contract, with
  `live_account_checked: false`. Use the separate local CLI for actual saved
  account state. Never treat a model catalog as an entitlement check.
- This phase is local authentication and text inference, not `JANUS_AUTONOMY_FULL`.
  No models, credits, tokens, new recurring jobs, NAS or external organs are
  automatically affected.

## Run the local client

Use Python 3.12 or newer and a separately installed official Codex CLI.
Create a virtual environment in a reviewed checkout:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r mcp_gateway/requirements.txt
codex --version
.\.venv\Scripts\python.exe -m mcp_gateway.codex_bridge status
.\.venv\Scripts\python.exe -m mcp_gateway.codex_bridge login
```

On Unix use `python3 -m venv .venv` and `.venv/bin/python`. The runtime
checks the actual app-server protocol and effective tool configuration;
incompatible Codex versions fail closed. Do not weaken these checks.

`login` opens **Continue with ChatGPT** in the system browser. Select the
account/workspace and review permissions there. Do not paste codes, callback
URLs or credentials into chat. Eligible inference uses the selected ChatGPT
plan; review plan/credit limits in [ChatGPT Settings → Usage](https://chatgpt.com/settings/usage).
No API key or client secret is used by this public-client flow.

Default private state is `%LOCALAPPDATA%\JANUS\CodexBridge` on Windows and
`$XDG_STATE_HOME/janus-codex-bridge` (fallback `~/.local/state/janus-codex-bridge`)
on Unix. `--state-dir PATH` must precede the subcommand and must identify a
dedicated directory outside source control, separate from ordinary Codex or
JANUS Habitat state. Windows uses CurrentUser DPAPI and restricted ACLs; Unix
uses owner-only permissions. Atomic writes and process locking serialize
rotating refresh tokens. The same OS user's other processes remain trusted.

```powershell
python -m mcp_gateway.codex_bridge accounts
python -m mcp_gateway.codex_bridge models
python -m mcp_gateway.codex_bridge select PROFILE_ID
python -m mcp_gateway.codex_bridge login --profile PROFILE_ID
python -m mcp_gateway.codex_bridge logout --profile PROFILE_ID
```

Account registrations remain separate even if their emails match. Returning
login reuses the issued client ID and stable host ID. Logout attempts remote
session revocation and clears local tokens while retaining the registration.
Check `remote_revocation_confirmed`; if false, disconnect the app in ChatGPT
Settings. Local `status` is a snapshot and makes no network or inference call.

`models` fetches the current account-specific OpenAI model choices. The
app-server's bundled catalog can contain an unavailable model; it is not
account entitlement evidence. Probe the exact executable/model locally before
authorizing one model request:

```powershell
python -m mcp_gateway.codex_bridge doctor --codex-binary PATH_TO_CODEX --model MODEL_ID
python -m mcp_gateway.codex_bridge smoke --codex-binary PATH_TO_CODEX --model MODEL_ID --allow-plan-usage
python -m mcp_gateway.codex_bridge ask --codex-binary PATH_TO_CODEX --model MODEL_ID --allow-plan-usage --prompt "Explain the JANUS gateway boundary."
python -m mcp_gateway.codex_bridge ask --codex-binary PATH_TO_CODEX --model MODEL_ID --allow-plan-usage --thread THREAD_ID --prompt "Expand the previous explanation."
```

Use `--model MODEL_ID` for an exact model. Otherwise the runtime selects the
default catalog entry; it never substitutes an explicitly chosen model.
Smoke succeeds only after a correlated successful `turn/completed` and the
expected `JANUS_BRIDGE_OK` text. Failed/interrupted turns, protocol mismatches,
unexpected tool/approval requests and timeouts fail closed.

The runtime exports the trusted executable's bundled model catalog locally and
reduces its execution metadata without changing model identifiers, reasoning
capabilities or account entitlements. The wire probe verifies the resulting
catalog rather than assuming a feature flag is effective.

The child receives a minimal environment, isolated per-account Codex home and
disposable working directory. Shell, MCP, hooks, plugins, apps, browser/computer
tools, web search and child agents are disabled in configuration. Configuration
alone is insufficient: a synthetic, loopback-only Responses probe inspects the
actual tool catalog for that exact binary/model before a credentialed child is
started. Unknown or executable tools fail closed. At most four schema-checked,
nonexecuting question/goal metadata tools may be present; any attempted tool
request or emitted tool item is rejected. No synthetic probe is forwarded to
OpenAI. `doctor` never reads credentials or makes an external model request.
GitHub credentials and the OAuth vault are excluded; the access token is passed
only in the child environment. Thread resume uses only this bridge's own
account-specific history. Renewal obtains the replacement token before starting
a fresh app-server process. This stage does not give a model code-write, commit,
PR, merge or deployment authority.

## Keep Habitat separate

The installed **JANUS Habitat** plugin is an append-only stdio service with
eight continuity tools. `mcp_gateway.server` is a separate HTTP MCP service;
updating this repository does not upgrade Habitat's tool inventory. Do not
replace Habitat's cache, journal or launch configuration with this server.

Run the repository gateway separately if needed:

```powershell
$env:JANUS_ALLOW_WRITES = '0'
$env:MCP_HOST = '127.0.0.1'
python -m mcp_gateway.server
```

Alternatively, run `python -m mcp_gateway.stdio_server` for a separately
registered local stdio gateway. This entry point forcibly removes inherited
GitHub/provider credentials and JANUS repository/private-binding overrides and
sets `JANUS_ALLOW_WRITES=0` before importing the server. It does not load the
SIWC client or token vault. Its startup can be managed by the local Codex host.

The HTTP default is now loopback (`127.0.0.1:8000/mcp`). Register this endpoint
separately. A network deployment needs its own tested transport authentication;
explicitly set `MCP_HOST=0.0.0.0` only for an intentionally secured deployment.

## Verify and roll back

```text
python -m unittest discover -s mcp_gateway/tests -v
python -m compileall -q mcp_gateway
python -c "from mcp_gateway.server import mcp; print(type(mcp).__name__)"
python -m mcp_gateway.codex_bridge --help
```

CI runs offline tests on Linux and Windows with synthetic credentials and fake
local processes. No test logs in or invokes a model. Rollback: stop this client,
use `logout` to revoke its session, then return to the prior checkout or revert
the reviewed PR. A Git rollback does not revoke OAuth. Do not reset Habitat.
Preserve private state until remote revocation is confirmed.

## References

- Codex Cloud: https://learn.chatgpt.com/docs/environments/cloud-environments
- Sessions and renewal: https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions
- Open-source Sign in with ChatGPT: https://developers.openai.com/siwc/token-sharing-open-source
- Dynamic registration / PKCE: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
- Official Codex app-server settings: https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server
- Preview limitations: https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
