"""JANUS Trait Immune Spiral R2.1: longitudinal behavioural drift gate."""
from dataclasses import dataclass
from math import sqrt

@dataclass(frozen=True)
class SpiralTurn:
    turn_id: str
    lineage_id: str
    parent_turn_id: str
    scores: dict
    constraints: tuple = ()

@dataclass(frozen=True)
class SpiralPolicy:
    local_rms_max: float = 0.10
    ancestral_rms_max: float = 0.15
    rollback_on_fail: bool = True

class TraitImmuneSpiral:
    @staticmethod
    def rms(left, right):
        if set(left) != set(right) or not left:
            return 1.0
        ds = [float(right[k]) - float(left[k]) for k in sorted(left)]
        return sqrt(sum(d*d for d in ds) / len(ds))

    @classmethod
    def evaluate(cls, ancestor, previous, candidate, policy=SpiralPolicy()):
        reasons = []
        if candidate.lineage_id != previous.lineage_id or candidate.lineage_id != ancestor.lineage_id:
            reasons.append("LINEAGE_MISMATCH")
        if candidate.parent_turn_id != previous.turn_id:
            reasons.append("BROKEN_SPIRAL_ANCESTRY")
        local = cls.rms(previous.scores, candidate.scores)
        ancestral = cls.rms(ancestor.scores, candidate.scores)
        if local > policy.local_rms_max:
            reasons.append("LOCAL_TURN_DRIFT")
        if ancestral > policy.ancestral_rms_max:
            reasons.append("CUMULATIVE_ANCESTRAL_DRIFT")
        if candidate.scores == previous.scores and set(candidate.constraints) <= set(previous.constraints):
            reasons.append("IDENTICAL_STATE_PLATEAU")
        allowed = not reasons
        return {
            "verdict": "PASS_SPIRAL_ASCEND" if allowed else "HOLD_OR_ROLLBACK",
            "allowed": allowed,
            "local_rms": local,
            "ancestral_rms": ancestral,
            "reasons": reasons,
            "rollback_required": (not allowed and policy.rollback_on_fail),
            "replay_affected_downstream_edges": not allowed,
        }
