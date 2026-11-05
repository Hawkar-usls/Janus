"""Separate, read-only local repository gateway; never replaces Habitat.

No user credential is accepted by this entry point. The public repository
gateway runs independently of the owner-operated SIWC client.
"""
from __future__ import annotations

import os


def main():
    # Remove ambient credential/binding overrides before importing the server.
    for key in tuple(os.environ):
        if key in {"GITHUB_TOKEN", "GH_TOKEN", "ACCESS_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"} \
                or key.startswith("JANUS_"):
            os.environ.pop(key, None)
    os.environ["JANUS_ALLOW_WRITES"] = "0"
    from .server import mcp
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
