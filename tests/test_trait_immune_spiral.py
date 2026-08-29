from trait_immune_spiral import SpiralTurn, TraitImmuneSpiral

def turn(i, scores, constraints=()):
    return SpiralTurn(str(i), "JANUS:IMMUNE", str(i-1) if i else "ROOT", scores, constraints)

def test_small_local_drift_can_fail_ancestral_drift():
    a = turn(0, {"x": 0.50, "y": 0.50})
    b = turn(1, {"x": 0.58, "y": 0.58}, ("c1",))
    c = turn(2, {"x": 0.66, "y": 0.66}, ("c1", "c2"))
    v = TraitImmuneSpiral.evaluate(a, b, c)
    assert v["local_rms"] < 0.10
    assert v["ancestral_rms"] > 0.15
    assert "CUMULATIVE_ANCESTRAL_DRIFT" in v["reasons"]
    assert v["rollback_required"]

def test_identical_state_is_plateau_not_ascent():
    a = turn(0, {"x": 0.5})
    b = turn(1, {"x": 0.5}, ("known",))
    c = turn(2, {"x": 0.5}, ("known",))
    v = TraitImmuneSpiral.evaluate(a, b, c)
    assert "IDENTICAL_STATE_PLATEAU" in v["reasons"]
    assert not v["allowed"]

def test_new_constraint_allows_stable_ascent():
    a = turn(0, {"x": 0.5})
    b = turn(1, {"x": 0.5}, ("known",))
    c = turn(2, {"x": 0.5}, ("known", "new"))
    v = TraitImmuneSpiral.evaluate(a, b, c)
    assert v["verdict"] == "PASS_SPIRAL_ASCEND"
