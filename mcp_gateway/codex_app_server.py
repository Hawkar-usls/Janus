"""Codex app-server integration contract for owner-authorized ChatGPT plan usage.

This public MCP contract never opens a token vault or launches a subprocess.
The separate, explicitly invoked local client in codex_bridge owns OAuth and
the isolated inference runtime. Its account state is not queried by this tool.
"""
from __future__ import annotations


class CodexProtocolError(ValueError):
    pass


def app_server_argv() -> list[str]:
    """Official stdio / Responses provider configuration (no credentials)."""
    return [
        "codex", "app-server", "--listen", "stdio://",
        "-c", 'model_provider="openai_chatgpt_plan"',
        "-c", 'model_providers.openai_chatgpt_plan.name="ChatGPT plan"',
        "-c", 'model_providers.openai_chatgpt_plan.base_url="https://api.openai.com/v1"',
        "-c", 'model_providers.openai_chatgpt_plan.env_key="ACCESS_TOKEN"',
        "-c", 'model_providers.openai_chatgpt_plan.wire_api="responses"',
        "-c", 'model_providers.openai_chatgpt_plan.requires_openai_auth=false',
        "-c", 'model_providers.openai_chatgpt_plan.supports_websockets=false',
    ]


def validate_turn_completion(event: dict) -> dict[str, object]:
    """Fail closed: only an actually completed turn is a successful response.

    Calling code must additionally validate source pins, allowed output scope
    and tests before trusting any generated patch. This method does not
    authorize committing, deploying, or merging.
    """
    if not isinstance(event, dict) or event.get("method") != "turn/completed":
        raise CodexProtocolError("CODEX_TURN_COMPLETION_NOT_OBSERVED")
    params = event.get("params")
    turn = params.get("turn") if isinstance(params, dict) else None
    if not isinstance(turn, dict) or turn.get("status") != "completed":
        raise CodexProtocolError("CODEX_TURN_NOT_SUCCESSFUL")
    thread_id = params.get("threadId")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise CodexProtocolError("CODEX_THREAD_ID_UNVERIFIED")
    return {
        "status": "MODEL_TURN_COMPLETED_ONLY",
        "thread_id": thread_id,
        "github_patch_verified": False,
        "github_pr_created": False,
        "auto_merge": False,
        "production_authority_delta": 0,
    }


def app_server_readiness() -> dict[str, object]:
    """Safe, public integration status without account or secret data."""
    return {
        "schema": "janus.codex.siwc_app_server_plan.v1",
        "status": "AUTHENTICATION_NOT_CONNECTED",
        "status_source": "static_contract_not_account_probe",
        "live_account_checked": False,
        "local_client_module": "mcp_gateway.codex_bridge",
        "local_status_command": "python -m mcp_gateway.codex_bridge status",
        "mode": "SIGN_IN_WITH_CHATGPT_USER_CONSENT",
        "requires_openai_oidc_pkce_and_jwks_validation": True,
        "requires_chatgpt_plan_usage_scope": "chatgpt.tokens.use.direct",
        "requires_per_host_stable_id": True,
        "requires_private_refresh_token_rotation": True,
        "token_transport": "CHILD_ENV_ACCESS_TOKEN_ONLY_AFTER_VALIDATED_CONSENT",
        "app_server_argv": app_server_argv(),
        "github_repository": "Hawkar-usls/Janus",
        "chat_history_access": False,
        "oauth_token_accessed": False,
        "codex_app_server_started": False,
        "production_authority_delta": 0,
    }
