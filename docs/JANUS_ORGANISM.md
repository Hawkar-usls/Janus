# JANUS Federated Organism v1.1

JANUS is treated as a **federated organism**, not as one repository with ambient authority over every other repository.

The canonical machine-readable map is [`organism/JANUS_ORGANISM_v1.json`](../organism/JANUS_ORGANISM_v1.json).

## Body plan

```text
                                      HUMAN
                                        |
                                 operator_hands
                               Terminal for Janus
                                        |
                                  gateway / MCP
                                        |
       +--------------------------------+--------------------------------+
       |                                |                                |
 left_context                    guardian_cortex                 right_association
    HRain                           Demi_Head                         iNaiHR
       \                                |                                /
        +-------------------------------+-------------------------------+
                                        |
       +----------------------+---------+---------+----------------------+
       |                      |                   |                      |
 causal_witness          proof_spine        measurement_bench       orchestrator
     AIFC              Fundamentum          janus-io-public          Demiurge
       |                      |                   |                      |
       +----------------------+---------+---------+----------------------+
                                        |
           +----------------------------+----------------------------+
           |                            |                            |
      anomaly_lab                  observatory                hypothesis_metabolism
         TOPA                    Janus-Cosmos                    janus-lapis
           |                            |                            |
           +----------------------------+----------------------------+
                                        |
                              simulation_habitat
                                Janus_Genesis

 WORLD -> private somatosensory slot -> sensorimotor mesh -> cognition
 cognition -> The-Voice-of-Janus -> Echo-Pyramid -> WORLD

 Domain labs: SCOBY-D0 / Fast-CAT-SHAiTan
 Social membrane: janus-first-followers-club-
 Long-term memory: janus-meta-registry
```

## Membership classes

- **CORE** — reasoning, provenance, causality, measurement, orchestration and primary interaction surfaces.
- **INTERFACE_PHYSICAL** — bounded human/social or physical output surfaces.
- **DOMAIN_LAB** — specialized experimental laboratories whose authority remains local to their domain.
- **PRIVATE_GATED** — organism slots whose real repository locator is not published and is unavailable to default MCP routing.

Membership is not a promotion in scientific authority.

## Core roles

- **Janus** — nervous-system gateway and MCP surface.
- **janus-meta-registry** — long-term memory, provenance, corrections and supersession.
- **Janus-Fundamentum** — scoped formal proof/falsification spine.
- **AIFC** — causal witness: precommitment, causal ordering, witness quorum, entropy evidence and fail-closed evidence grades.
- **Demi_Head** — guardian/review cortex and disagreement-preserving HOLD surface.
- **Janus-Demiurge** — spiral agent control plane and Scout orchestration.
- **Terminal for Janus** — human-authorized operator hands. A command is not evidence and access is not ambient authority.
- **HRain** — structural-context / left-context surface.
- **iNaiHR** — associative-semantic / right-context surface. Association is not evidence.
- **janus-io-public** — proof-of-observation and measurement bench between telemetry and scientific interpretation.
- **janus-distributed-ai-swarm** — embedded sensorimotor mesh.
- **janus-lapis** — exploratory hypothesis ranking/metabolism.
- **aura-oracle-tg** — symbolic imagination with zero empirical authority.
- **TOPA** — anomaly triage and falsification packets.
- **Janus-Cosmos** — observational / signal-research organ.
- **Janus_Genesis** — persistent simulation/development habitat. Simulation traces never become world evidence by transport.

## Physical and interface surfaces

- **The-Voice-of-Janus** — deterministic voice/audio/DSP output contract.
- **Echo-Pyramid** — bounded physical audio actuator and device telemetry surface.
- **somatosensory_skin** — private-gated tactile/thermal/mechanical sensing slot. The public manifest contains no private repository name; a deployment must explicitly bind `JANUS_PRIVATE_SOMATOSENSORY_REPO`.
- **janus-first-followers-club-** — voluntary social membrane / public handshake surface. Any external effect remains human-authorized.

## Domain laboratories

- **SCOBY-D0** — bacterial-cellulose/materials laboratory; field readiness requires its own frozen empirical gates.
- **Fast-CAT-SHAiTan** — feline facial-timing laboratory; its timing/review authority stays within the feline experiment scope.

Domain-lab methods can inspire reusable patterns, but joining the organism does not make a domain-specific result universal.

## External dependencies, not organs

`Linear-A-decipherment-programme` remains classified as an **external research instrument** because its upstream academic authorship/provenance is independent. JANUS may use it as a tool without reclassifying the upstream programme as a JANUS organ.

## Constitutional boundary

```text
REPOSITORY_MEMBERSHIP != TRUTH
MODEL_OUTPUT != EVIDENCE
SYMBOLIC_OUTPUT != EMPIRICAL_EVIDENCE
RANKING != VALIDATION
DISPATCH != AUTHORITY
COMMAND != EVIDENCE
MULTIPLE_AGENTS != INDEPENDENT_SOURCES
HASH_OR_SIGNATURE != TRUTH_OF_CONTENT
SIMULATION != WORLD_EVIDENCE
MEASUREMENT != MECHANISM
SENSOR_OUTPUT != CALIBRATED_MEASUREMENT
NEGATIVE_RESULTS_AND_DISAGREEMENT_ARE_PRESERVED
WRITE_BACK_DEFAULT = DENY
```

Each organ preserves its own maturity, provenance and claim boundary. Membership never grants another repository's evidence class, credentials, production status or external-effect authority.

## Typical evidence path

```text
human / sensor / observation
  -> structural + associative context where useful
  -> measurement / causal witness
  -> TOPA competing hypotheses
  -> guardian source-root and disagreement review
  -> formal attack where formalizable
  -> domain experiment or simulation where appropriate
  -> memory receipt / correction / negative result
  -> another spiral only when a real open gate remains
```

No stage silently promotes the previous stage's output.

## MCP routing

`janus.search_organ` and `janus.read_organ` accept typed organ keys from an explicit public allowlist. Arbitrary `owner/repository` input is rejected.

Private organs are fail-closed. `somatosensory_skin` cannot resolve unless the deployment explicitly binds its private locator through `JANUS_PRIVATE_SOMATOSENSORY_REPO` and provides credentials that can read that repository. The real private repository name is intentionally absent from the public manifest and public source allowlist.

## Writes

The MCP gateway remains read-first. Existing write tools write only to the dedicated `Hawkar-usls/Janus:mcp-checkpoints` branch when explicitly enabled. Organism membership does not authorize writes to member repositories.

## Expansion rule

A repository should become a distinct organ only when it supplies a genuinely distinct system function. Upstream mirrors, libraries, compatibility snapshots and ordinary dependencies should be modeled as dependencies, tissues or tools rather than receiving artificial organ-level authority.
