from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .core import sanitize_repo_path


class GitHubAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class GitHubConfig:
    token: str | None
    api_base: str = "https://api.github.com"
    user_agent: str = "janus-mcp-gateway/0.1"
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "GitHubConfig":
        return cls(token=os.getenv("GITHUB_TOKEN") or None)


class GitHubAPI:
    def __init__(self, config: GitHubConfig | None = None):
        self.config = config or GitHubConfig.from_env()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = path if path.startswith("https://") else f"{self.config.api_base.rstrip('/')}/{path.lstrip('/')}"
        data = None
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.config.user_agent,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
            raise GitHubAPIError(f"GitHub API {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise GitHubAPIError(f"GitHub API connection failed: {exc.reason}") from exc

    def get_file(self, repo: str, path: str, ref: str = "main") -> dict[str, Any]:
        path = sanitize_repo_path(path)
        quoted = urllib.parse.quote(path, safe="/")
        ref_q = urllib.parse.quote(ref, safe="")
        item = self._request("GET", f"repos/{repo}/contents/{quoted}?ref={ref_q}")
        if not isinstance(item, dict) or item.get("type") != "file":
            raise GitHubAPIError("requested path is not a file")
        content = base64.b64decode(item.get("content", "")).decode("utf-8", errors="replace")
        return {
            "repo": repo,
            "path": path,
            "ref": ref,
            "sha": item.get("sha"),
            "size": item.get("size"),
            "html_url": item.get("html_url"),
            "content": content,
        }

    def search_code(self, repo: str, query: str, limit: int = 10, path_prefix: str | None = None) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        limit = max(1, min(int(limit), 25))
        parts = [query, f"repo:{repo}"]
        if path_prefix:
            parts.append(f"path:{sanitize_repo_path(path_prefix)}")
        q = urllib.parse.quote(" ".join(parts), safe="")
        result = self._request("GET", f"search/code?q={q}&per_page={limit}")
        items = result.get("items", []) if isinstance(result, dict) else []
        return [
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "sha": item.get("sha"),
                "html_url": item.get("html_url"),
                "repository": item.get("repository", {}).get("full_name"),
            }
            for item in items[:limit]
        ]

    def put_file(
        self,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str,
        sha: str | None = None,
    ) -> dict[str, Any]:
        path = sanitize_repo_path(path)
        quoted = urllib.parse.quote(path, safe="/")
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha
        result = self._request("PUT", f"repos/{repo}/contents/{quoted}", payload)
        return {
            "repo": repo,
            "path": path,
            "branch": branch,
            "commit_sha": (result or {}).get("commit", {}).get("sha"),
            "html_url": (result or {}).get("content", {}).get("html_url"),
        }
