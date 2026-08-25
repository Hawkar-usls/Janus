# JANUS Repository Ecology v1.3

JANUS is a federated organism, not a monorepo and not a claim that every repository has equal authority.

v1.3 adds a repository-ecology layer above the canonical organ topology and below/alongside the 5D Spiral execution constitution.

## Why this layer exists

The account contains active organs, child projects, private evidence stores, games, interfaces, legacy prototypes, upstream working copies and external research instruments. Treating all of them as peer organs would erase provenance and create false authority. Treating only core organs as part of JANUS would lose lineage and useful tissue.

The ecology therefore uses six bounded classes:

- `ORGAN` — canonical typed organ from `JANUS_ORGANISM_v1.json`.
- `SUBTISSUE` — bounded child repository under one organ.
- `PRIVATE_SUBTISSUE` — fail-closed child repository whose concrete locator is never published by the public manifest.
- `TOOLCHAIN_DEPENDENCY` — upstream-derived working copy or infrastructure dependency.
- `EXTERNAL_RESEARCH_INSTRUMENT` — independent scholarly/data instrument with preserved external provenance.
- `LEGACY_ARCHIVE` — historical JANUS-native prototype kept for lineage/replay, not assumed active.

## New private subtissues

### Private measurement substrate

Parent organ: `measurement_bench` (`janus-io-public`).

Purpose: private raw evidence, internal experiment state, run archives and curated-export candidates.

Public binding uses only:

```text
JANUS_PRIVATE_MEASUREMENT_REPO
```

Boundary:

```text
RAW_PRIVATE_EVIDENCE != PUBLIC_REVIEWED_EVIDENCE
ACCEPTED_SHARE_TELEMETRY != SHA256_BREAK
SUBTISSUE_PARENT != AUTHORITY_INHERITANCE
```

### Private Genesis world

Parent organ: `simulation_habitat` (`Janus_Genesis`).

Purpose: private persistent simulation/game world and player/habitat handoff.

Public binding uses only:

```text
JANUS_PRIVATE_GENESIS_WORLD_REPO
```

Boundary:

```text
SIMULATION != WORLD_EVIDENCE
GAME_STATE != PHYSICAL_WORLD_STATE
HABITAT_HANDOFF != COMMAND_AUTHORITY
```

## Public subtissues

The following JANUS-native repositories are explicitly subordinate to `simulation_habitat` rather than promoted to peer organs:

- BFain — command/control visualization simulation.
- DIVINE_REALM — narrative/symbolic world.
- ATOM ELITE — embedded retro game world.
- ATOM RPG — embedded RPG experiment.

Their outputs can enter a Spiral turn as simulation traces, countermodels, interface states or creative world states, but never as independent empirical evidence.

## Toolchain dependencies

Upstream-derived repositories are registered centrally but are not modified merely to add JANUS branding. This preserves clean provenance and avoids implying base authorship. Toolchain use does not promote evidence.

The ML repository named `tranception` remains an upstream-derived dependency and is explicitly distinct from the JANUS whole-organism Tranception reasoning operator.

## External research instruments

Linear A repositories are treated as external research instruments where original scholarly/data provenance remains independent. JANUS may analyze or route them, but does not absorb upstream authorship.

## Legacy

Legacy prototypes remain addressable in ecology for historical replay and lineage. Presence in ecology does not mean the capability is active, validated or recommended for deployment.

## Spiral integration

Repository ecology feeds the existing v1.2 execution law:

```text
repository/subtissue/dependency
    -> typed parent organ
    -> 5D Spiral / Tranception
    -> checkpoint
    -> ASCEND or PLATEAU
```

No transport creates authority:

```text
AUTHORITY_DELTA_ON_TRANSPORT = 0
```

A repository becomes a new peer `ORGAN` only when it has a genuinely distinct function, typed inputs/outputs, an explicit claim boundary and a reason that function cannot remain safely subordinate to an existing organ.
