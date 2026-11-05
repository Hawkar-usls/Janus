# JANUS account-based model gateway

This local interface reuses the owner's authenticated ChatGPT and Google
accounts without asking them to copy API keys or access/refresh tokens.
Provider clients still authenticate internally. There is no unauthenticated
access to a paid account and no unlimited quota claim.

## Routes

| Provider | Supported account route | Verification on 2026-11-05 |
|---|---|---|
| ChatGPT | Reviewed local SIWC + Codex app-server client from PR #49 | Real completed gpt-6.1-sol turn and resume |
| Google | Official Antigravity CLI 1.5.1, Google OAuth + OS keyring | Real gemini-3.8-flash-low completed turn |
| DeepSeek | Official API requires an API key; web UI is a separate host-operated surface | Disabled in the keyless automatic pool |

The Google account catalog also exposed Claude Sonnet 4.6, Claude Opus 4.6 and
GPT-OSS 120B. A catalog entry alone is not a successful inference/entitlement
test for each model. Account catalogs and quotas can change.

The current Google route is Antigravity, not the legacy Gemini CLI consumer
login. Google's newer deprecation notice states that personal/free/Pro/Ultra
Google login on Gemini CLI ended on June 18, 2026. Older Gemini CLI auth pages
can still describe the previous flow; use the current notice and Antigravity
authentication docs linked below.

## Local commands

```text
python -m mcp_gateway.account_gateway --help
python -m mcp_gateway.account_gateway --chatgpt-state-dir PRIVATE_DIR --codex-binary CODEX_EXE --google-binary AGY_EXE status
python -m mcp_gateway.account_gateway --google-binary AGY_EXE models --provider google
python -m mcp_gateway.account_gateway --google-binary AGY_EXE smoke --provider google --model gemini-3.8-flash-low --allow-plan-usage
```

All global path options precede the subcommand. Paths are trusted local-owner
configuration and are never exposed through MCP or supplied by model output.
The saved Windows launcher supplies them automatically.

Use `ask --provider chatgpt|google --prompt TEXT --allow-plan-usage` for a
single chosen provider. Use `--model EXACT_SLUG` to pin a model; the gateway
does not substitute an explicitly selected model. `--thread ID` resumes that
provider's conversation and must be used with an explicit provider.

`ask --provider auto --prompt TEXT --allow-plan-usage` first uses ChatGPT and
then retries the same text through Google only for recognized quota/rate-limit
errors. This explicitly authorizes sending that request to either provider.
Authentication failures, protocol/configuration errors and arbitrary exceptions
do not cause cross-provider fallback. `auto` rejects exact model/thread pins,
so a ChatGPT conversation ID is never reinterpreted as a Google conversation.
Provider defaults apply in auto mode. No paid API or purchased credits are
automatically configured as fallback.

Status is a local snapshot. `models` queries the official client's account
catalog. `smoke` and `ask` perform model calls and require the explicit usage
flag. The gateway does not create accounts, buy subscriptions, or increase
account limits. Existing provider-plan/credit settings still apply.

## Google client installation and sign-in

Download Antigravity CLI from its official vendor installer. The local setup
used the vendor's Windows release manifest and verified the binary's SHA-512
before running it. There is no global PATH modification required.

Run `agy` interactively once and choose Google OAuth. Complete Google's browser
sign-in yourself. If the browser displays a one-time authorization code, paste
it directly into the official CLI, never into a model chat. Successful login is
kept by Google's unified native keyring; no JANUS token copier is involved.
`agy models` checks whether this sign-in can fetch the catalog.

The SDK's API-key/Vertex configuration is a different route. This integration
does not enable it and excludes GEMINI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS,
cloud-project overrides, ChatGPT tokens and GitHub tokens from the child env.

Every Google inference starts in a disposable directory with a session-only
deny policy for file reads/writes, commands, MCP and browser/network tools.
It uses plan mode and disables slash-command expansion. It never uses
`--dangerously-skip-permissions`. The subprocess has a bounded lifetime/output;
only a SUCCESS JSON envelope with text and a conversation ID is accepted.
Provider diagnostic text is classified into fixed codes, not copied to chat.
The Google CLI and the same OS user's other processes remain trusted runtime
components; these settings do not sandbox a compromised binary.

## DeepSeek and local models

DeepSeek's documented programmatic route requests an API key. Its Harness also
asks for that key. A logged-in chat.deepseek.com browser is not equivalent to
an authenticated public API. Browser interaction may be performed separately
by the host when authorized; it is not advertised as an unattended keyless
provider in this pool. No cookies/session secrets are extracted.

Ollama was detected on the Windows host but no installed models were listed.
No large model download, Habitat capability change or local inference adapter
was automatically enabled by this account integration.

## Validation and rollback

Run `python -m unittest discover -s mcp_gateway/tests -v` and
`python -m compileall -q mcp_gateway`. The new offline tests exercise routing,
quota-only fallback, no credential inheritance, denial policy delivery to a
real child, malformed/failed completion envelopes and unsupported DeepSeek.

This change is stacked on the SIWC bridge branch. Review/merge PR #49 before
accepting the provider-pool change. It does not edit that existing checkout,
its pinned launcher or Habitat journal. To roll back, stop using the new
account gateway and return to the ChatGPT-only launcher. To revoke Google
access, use `/logout` in the official Antigravity client or Google account
settings. Source removal alone does not revoke a saved provider session.

## Primary references

- [Google consumer migration notice](https://developers.google.com/gemini-code-assist/docs/deprecations/code-assist-individuals)
- [Antigravity authentication](https://antigravity.google/docs/install/)
- [Antigravity headless interface](https://antigravity.google/docs/cli/headless/)
- [Antigravity permissions](https://antigravity.google/docs/permissions/)
- [Antigravity download](https://antigravity.google/download)
- [DeepSeek API authentication](https://api-docs.deepseek.com/)
- [DeepSeek Harness model setup](https://deepseek-harness.github.io/deepseek-harness/en/guide/quickstart)
