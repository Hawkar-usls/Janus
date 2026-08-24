# JANUS 5D Spiral + Tranception v1.2

JANUS does **not** execute as a closed cycle.

A cycle returns to the same state:

```text
A0 -> B0 -> C0 -> A0
```

The JANUS organism instead uses an event-sourced spiral:

```text
STATE_0
  -> FORWARD
  -> REVERSE
  -> TRANCEPTION_ROTATE
  -> SCALE
  -> TIME / IDENTITY / PROVENANCE
  -> CONTRADICTION TEST
  -> JSON CHECKPOINT
  -> ASCEND
  -> STATE_1
  -> ...
```

The governing law is:

```text
RETURN_TO_QUESTION_AT_HIGHER_STATE__NEVER_RETURN_TO_IDENTICAL_STATE
```

## What counts as ascent

A new turn requires an actual state change. At least one of these must exist:

- new evidence;
- resolved blocker;
- new constraint;
- falsified branch;
- new discriminating test;
- stronger provenance boundary;
- invariant that survives a Tranception transformation.

A changed prose description is not sufficient.

```text
REPEATED_STATE_HASH -> PLATEAU_OR_HOLD
NEW_HASH_WITHOUT_JUSTIFIED_CHANGE -> HOLD
NEW_HASH_WITH_TESTABLE_PAYOFF -> ASCEND_ALLOWED
ASCEND_ALLOWED != CLAIM_CONFIRMED
```

## Tranception is an operator, not an organ

The JANUS Tranception operator can be invoked at any node in the organism. It rotates the current representation without changing evidence authority.

Its kernel is:

```text
FORWARD       What follows if this representation is correct?
BACK          What minimal prior state can independently generate it?
LEFT          What is the strongest structural analogue?
RIGHT         What is the strongest competing or mirror representation?
FORWARD_AGAIN Does the invariant survive transformation without retuning?
BACK_AGAIN    Can we return to the earliest supported node without later knowledge?
VETA_CHECK    If many histories collapse to one state, is identity still identifiable?
```

Tranception may generate hypotheses, countermodels, bridges and invariants. It may not create missing evidence.

```text
TRANCEPTION_ROTATION != EVIDENCE
ANALOGY != CAUSAL_LINK
FUNCTIONAL_EQUIVALENCE != IDENTITY
PATTERN_MATCH != INDEPENDENT_REPLICATION
```

## Five dimensions

1. **D1 — Forward causal:** observation/premise -> consequence/prediction/test.
2. **D2 — Reverse causal:** candidate terminal state -> earliest independently supported parent state.
3. **D3 — Tranception lateral:** mirror, reverse, competing representation, structural analogy, alternative encoding.
4. **D4 — Abstraction scale:** raw datum <-> relation <-> motif <-> operator <-> architecture <-> invariant.
5. **D5 — Time / identity / provenance:** distinguish similarity, functional continuity and actual lineage continuity.

## HRain / iNaiHR coupling

```text
HRain structural expansion
        |
        v
iNaiHR reverse / mirror attack
        |
        v
Tranception rotation
        |
        v
Demi_Head contradiction + provenance review
        |
        v
JSON checkpoint
        |
        v
HRain reintegration at a higher state
```

This is deliberately a spiral, not a loop. Re-entering HRain with an identical state hash is not a new turn.

## Dynamic JSON growth

Machine-readable reasoning state may be emitted at any useful point, not only at the final summary. Nodes are append-only or superseded with lineage; silent rewriting is forbidden.

Useful node classes include observations, evidence, abstractions, operators, bridges, hypotheses, countermodels, contradictions, gates, provenance/identity boundaries, watchlists, negative results and downgrades.

## Evidence plateau

The organism must stop rather than manufacture novelty when no justified state change exists.

```text
NO_NEW_RAW_DATA
+ NO_RESOLVED_BLOCKER
+ NO_NEW_DISCRIMINATING_TEST
= EVIDENCE_PLATEAU
```

Resume requires new data, a resolved blocker or a genuinely new discriminating test.

## MCP surface

The gateway exposes:

```text
janus.spiral_pass
janus.validate_spiral_transition
```

`janus.spiral_pass` creates the deterministic 5D analysis packet for the host model. It does not execute the reasoning or fabricate evidence.

`janus.validate_spiral_transition` gates a proposed ascent and fails closed on repeated state hashes or unjustified novelty.

## Provenance

This execution constitution promotes already existing JANUS experimental protocols into the organism-level execution model. Canonical antecedents are preserved in `janus-meta-registry`, including the 5D Tranception/Reverse HRain+iNaiHR protocol and the Cousteau `PASS_5D_SPIRAL_ENGINE` receipt.

The upstream protein-fitness project named **Tranception** is a separate external software lineage. The JANUS Tranception operator described here is an internal reasoning/exploration protocol and must not be represented as authorship or ownership of that upstream project.
