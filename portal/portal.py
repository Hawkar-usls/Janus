from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "janus.portal.manifest.v1"
RECEIPT_SCHEMA = "janus.portal.route_receipt.v1"
ALLOWED_ROUTE_PREFIXES = ("repo:", "service:", "catalog:")
DESTINATION_ID = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


class PortalError(ValueError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise PortalError("Unsupported portal manifest schema")
    if manifest.get("mode") != "REFERENCE_ROUTE_DISCOVERY_ONLY":
        raise PortalError("Portal mode must remain reference-only")

    languages = manifest.get("supported_presentation_languages")
    if not isinstance(languages, list) or not languages or len(languages) != len(set(languages)):
        raise PortalError("supported_presentation_languages must be a non-empty unique list")

    destinations = manifest.get("destinations")
    if not isinstance(destinations, list) or not destinations:
        raise PortalError("Portal requires at least one destination")

    seen: set[str] = set()
    for destination in destinations:
        if not isinstance(destination, dict):
            raise PortalError("Each destination must be an object")
        destination_id = destination.get("id")
        if not isinstance(destination_id, str) or not DESTINATION_ID.fullmatch(destination_id):
            raise PortalError(f"Invalid destination id: {destination_id!r}")
        if destination_id in seen:
            raise PortalError(f"Duplicate destination id: {destination_id}")
        seen.add(destination_id)

        route_ref = destination.get("route_ref")
        if not isinstance(route_ref, str) or not route_ref.startswith(ALLOWED_ROUTE_PREFIXES):
            raise PortalError(f"Destination {destination_id} has unsupported route_ref")
        if "://" in route_ref:
            raise PortalError(f"Destination {destination_id} attempts arbitrary URL routing")
        if destination.get("portal_grants_effect_authority") is not False:
            raise PortalError(f"Destination {destination_id} must not inherit effect authority")

    default_destination = manifest.get("default_destination")
    if default_destination not in seen:
        raise PortalError("default_destination must name a declared destination")


def destination_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in manifest["destinations"]}


def list_destinations(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "janus.portal.destination_list.v1",
        "portal_id": manifest["portal_id"],
        "mode": manifest["mode"],
        "default_destination": manifest["default_destination"],
        "destinations": [
            {
                "id": item["id"],
                "title": item["title"],
                "class": item["class"],
                "effect_scope": item["effect_scope"],
            }
            for item in manifest["destinations"]
        ],
        "effect_authorized": False,
        "authority_delta": 0,
        "mass_effect_budget_delta": 0,
    }


def inspect_destination(manifest: dict[str, Any], destination_id: str) -> dict[str, Any]:
    destination = destination_map(manifest).get(destination_id)
    if destination is None:
        return _not_found_receipt(manifest, destination_id, None)
    return {
        "schema": "janus.portal.destination_inspection.v1",
        "portal_id": manifest["portal_id"],
        "destination": destination,
        "effect_authorized": False,
        "authority_delta": 0,
        "mass_effect_budget_delta": 0,
        "laws": manifest["laws"],
    }


def resolve_route(
    manifest: dict[str, Any],
    destination_id: str | None = None,
    presentation_language: str | None = None,
) -> dict[str, Any]:
    requested = destination_id or manifest["default_destination"]
    destinations = destination_map(manifest)

    if presentation_language is not None and presentation_language not in manifest["supported_presentation_languages"]:
        return _not_found_receipt(manifest, requested, presentation_language, status="UNSUPPORTED_PRESENTATION_LANGUAGE")

    destination = destinations.get(requested)
    if destination is None:
        return _not_found_receipt(manifest, requested, presentation_language)

    return {
        "schema": RECEIPT_SCHEMA,
        "portal_id": manifest["portal_id"],
        "status": "ROUTE_RESOLVED_REFERENCE_ONLY",
        "requested_destination": requested,
        "presentation_language": presentation_language,
        "route": {
            "destination_id": destination["id"],
            "destination_class": destination["class"],
            "route_ref": destination["route_ref"],
            "effect_scope": destination["effect_scope"],
        },
        "effect_authorized": False,
        "provider_realized": False,
        "permission_inferred": False,
        "truth_claim": "NOT_MADE",
        "evidence_state_mutated": False,
        "authority_delta": 0,
        "mass_effect_budget_delta": 0,
        "user_may_decline": True,
        "next_gate_owner": destination["id"],
        "laws": manifest["laws"],
    }


def decline_route(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "portal_id": manifest["portal_id"],
        "status": "ROUTE_DECLINED_VALID",
        "effect_authorized": False,
        "authority_delta": 0,
        "mass_effect_budget_delta": 0,
        "penalty": False,
    }


def _not_found_receipt(
    manifest: dict[str, Any],
    requested: str,
    presentation_language: str | None,
    status: str = "UNKNOWN_DESTINATION_FAIL_CLOSED",
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "portal_id": manifest["portal_id"],
        "status": status,
        "requested_destination": requested,
        "presentation_language": presentation_language,
        "route": None,
        "effect_authorized": False,
        "provider_realized": False,
        "permission_inferred": False,
        "truth_claim": "NOT_MADE",
        "evidence_state_mutated": False,
        "authority_delta": 0,
        "mass_effect_budget_delta": 0,
        "user_may_decline": True,
    }


def self_test(manifest: dict[str, Any]) -> dict[str, Any]:
    default = resolve_route(manifest)
    unknown = resolve_route(manifest, "UNDECLARED_WORLD")
    unsupported_language = resolve_route(manifest, "DEMIHEAD", "xx")
    declined = decline_route(manifest)

    bad_manifest = json.loads(json.dumps(manifest))
    bad_manifest["destinations"][0]["route_ref"] = "https://example.com/arbitrary"
    arbitrary_url_rejected = False
    try:
        validate_manifest(bad_manifest)
    except PortalError:
        arbitrary_url_rejected = True

    checks = {
        "default_is_demihead": default["route"]["destination_id"] == "DEMIHEAD",
        "route_has_no_effect_authority": default["effect_authorized"] is False,
        "unknown_fails_closed": unknown["status"] == "UNKNOWN_DESTINATION_FAIL_CLOSED" and unknown["route"] is None,
        "unsupported_language_fails_closed": unsupported_language["status"] == "UNSUPPORTED_PRESENTATION_LANGUAGE",
        "decline_is_valid": declined["status"] == "ROUTE_DECLINED_VALID" and declined["penalty"] is False,
        "arbitrary_url_rejected": arbitrary_url_rejected,
        "authority_delta_zero": default["authority_delta"] == 0,
        "mass_effect_delta_zero": default["mass_effect_budget_delta"] == 0,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"self_test": "PASS", "checks": checks}


def _render(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="JANUS First Portal deterministic reference resolver")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--list", action="store_true", dest="list_routes")
    parser.add_argument("--inspect")
    parser.add_argument("--resolve")
    parser.add_argument("--language", choices=["uk", "ru", "en"])
    parser.add_argument("--decline", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.self_test:
        payload = self_test(manifest)
    elif args.list_routes:
        payload = list_destinations(manifest)
    elif args.inspect:
        payload = inspect_destination(manifest, args.inspect)
    elif args.decline:
        payload = decline_route(manifest)
    else:
        payload = resolve_route(manifest, args.resolve, args.language)
    _render(payload, args.output)


if __name__ == "__main__":
    main()
