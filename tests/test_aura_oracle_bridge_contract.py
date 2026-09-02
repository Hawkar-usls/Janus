#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    link = json.loads((ROOT / ".janus/AURA_ORACLE_LINK.json").read_text(encoding="utf-8"))
    core = (ROOT / "janus_core.py").read_text(encoding="utf-8")
    organism = json.loads((ROOT / "organism/JANUS_ORGANISM_v1.json").read_text(encoding="utf-8"))

    assert link["organ_key"] == "symbolic_imagination"
    assert link["organ_repository"] == "Hawkar-usls/aura-oracle-tg"
    assert link["module_authority_lane"] == "BRANCH_AND_VERIFY"
    assert link["direct_main_write"] is False
    assert link["autonomous_merge"] is False
    assert link["authority_delta"] == 0

    member = organism["members"]["symbolic_imagination"]
    assert member["repo"] == "Hawkar-usls/aura-oracle-tg"
    assert member["authority"] == "ZERO_EMPIRICAL_AUTHORITY"
    assert member["evidence_authority"] is False

    required_routes = (
        "app.router.add_get('/api/get_user_state'",
        "app.router.add_post('/api/generate_cards'",
        "app.router.add_post('/api/interpret'",
    )
    for route in required_routes:
        assert route in core, f"missing JANUS AURA runtime route: {route}"

    assert "CREATE TABLE IF NOT EXISTS oracle_readings" in core
    assert "async def run_oracle_cards" in core
    assert "async def run_oracle_interpret" in core

    print("JANUS_AURA_ORACLE_BRIDGE=PASS")


if __name__ == "__main__":
    main()
