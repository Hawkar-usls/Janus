"""R2 lineage selector: demonstrate rules on synthetic assertions, NOT real model gain."""
import copy
import unittest
from autonomy.pillars import fitness as f


def synthetic_record():
    return {
      "schema":"janus.pillars.paired_fitness.v1",
      "parent_checkpoint_sha256":"a"*64,
      "candidate_checkpoint_sha256":"b"*64,
      "benchmark_sha256":"c"*64,
      "partition":"PUBLIC_INTERNAL_SANDBOX_NOT_GRADUATION",
      "training_on_benchmark":False,
      "external_independent_custodian":False,
      "parent_runtime_kind":"SAME_CPU_GITHUB_RUNNER_CLASS",
      "candidate_runtime_kind":"SAME_CPU_GITHUB_RUNNER_CLASS",
      "cases":[{"case_id":f"CASE_{i+1:02d}",
                "parent_correct":i<8,
                "candidate_correct":i<10,
                "parent_cpu_ms":100,
                "candidate_cpu_ms":110} for i in range(12)]}


class PairedFitness(unittest.TestCase):
    def test_promising_internal_never_graduates(self):
        result=f.examine(synthetic_record())
        self.assertEqual(result["status"],
            "INTERNAL_CANDIDATE_PROMISING__EXTERNAL_GRADUATION_HOLD")
        self.assertEqual(result["parent_correct"],8)
        self.assertEqual(result["candidate_correct"],10)
        self.assertEqual(result["newly_solved"],2)
        self.assertEqual(result["lost_previously_solved"],0)
        for k in ("may_replace_parent","may_promote_weights","may_spawn_organ",
                  "independent_verification_completed","training_weights_modified"):
            self.assertFalse(result[k],k)
        self.assertEqual(result["new_external_exam_batches_passed"],0)

    def test_one_lost_parent_case_rejects_candidate(self):
        x=synthetic_record()
        x["cases"][0]["candidate_correct"]=False
        result=f.examine(x)
        self.assertEqual(result["status"],"HOLD_INTERNAL_FUNCTIONAL_GAIN_OR_REGRESSION")
        self.assertEqual(result["lost_previously_solved"],1)

    def test_over_budget_cpu_rejects_even_higher_score(self):
        x=synthetic_record()
        for row in x["cases"]:row["candidate_cpu_ms"]=121
        self.assertEqual(f.examine(x)["status"],
                         "HOLD_INTERNAL_FUNCTIONAL_GAIN_OR_REGRESSION")

    def test_zero_strong_parent_cannot_be_used_as_fake_baseline(self):
        x=synthetic_record()
        for row in x["cases"]:row["parent_correct"]=False
        self.assertEqual(f.examine(x)["status"],
                         "HOLD_INTERNAL_FUNCTIONAL_GAIN_OR_REGRESSION")

    def test_external_independence_cannot_be_claimed_here(self):
        for key,value in (("external_independent_custodian",True),
                          ("training_on_benchmark",True),
                          ("partition","EXTERNAL_BLIND")):
            x=synthetic_record();x[key]=value
            with self.subTest(key=key),self.assertRaises(f.FitnessHold):
                f.examine(x)

    def test_identity_and_cost_requirements(self):
        for mutate in (
          lambda x:x["cases"][0].update(case_id=x["cases"][1]["case_id"]),
          lambda x:x["cases"][0].update(parent_correct=1),
          lambda x:x["cases"][0].update(candidate_cpu_ms=0),
          lambda x:x.update(candidate_checkpoint_sha256="a"*64),
          lambda x:x.update(candidate_runtime_kind="GPU_OR_UNVERIFIED"),
          lambda x:x["cases"].pop(),
        ):
            x=synthetic_record();mutate(x)
            with self.assertRaises(f.FitnessHold):
                f.examine(x)


if __name__=="__main__":
    unittest.main()
