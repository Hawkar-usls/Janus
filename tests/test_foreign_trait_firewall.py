# -*- coding: utf-8 -*-
import unittest

from foreign_trait_firewall import (
    ForeignTraitFirewall, LearningPermission, MemoryPlane, Provenance,
    SourceClass, TrainingSource,
)


class ForeignTraitFirewallTests(unittest.TestCase):
    def test_foreign_model_output_is_reference_only(self):
        d = ForeignTraitFirewall.admit_memory(Provenance(
            source_class=SourceClass.MODEL_GENERATED_FOREIGN,
            generator_model="foreign-model",
            memory_plane=MemoryPlane.IDENTITY,
            learning_permission=LearningPermission.IDENTITY_WRITE,
            approved=True,
        ))
        self.assertEqual(d.memory_plane, MemoryPlane.REFERENCE)
        self.assertFalse(d.identity_write_allowed)

    def test_unknown_provenance_is_quarantined(self):
        d = ForeignTraitFirewall.admit_memory(Provenance(source_class=SourceClass.UNKNOWN))
        self.assertEqual(d.memory_plane, MemoryPlane.QUARANTINE)

    def test_user_input_cannot_become_identity_automatically(self):
        d = ForeignTraitFirewall.admit_memory(Provenance(
            source_class=SourceClass.USER_SUPPLIED,
            memory_plane=MemoryPlane.IDENTITY,
            approved=True,
        ))
        self.assertEqual(d.memory_plane, MemoryPlane.REFERENCE)
        self.assertFalse(d.identity_write_allowed)

    def test_janus_identity_needs_owned_lineage_and_approval(self):
        denied = ForeignTraitFirewall.admit_memory(Provenance(
            source_class=SourceClass.JANUS_OWNED,
            lineage_id="JANUS:CORE",
            memory_plane=MemoryPlane.IDENTITY,
            approved=False,
        ))
        allowed = ForeignTraitFirewall.admit_memory(Provenance(
            source_class=SourceClass.JANUS_OWNED,
            lineage_id="JANUS:CORE",
            memory_plane=MemoryPlane.IDENTITY,
            approved=True,
        ))
        self.assertEqual(denied.memory_plane, MemoryPlane.QUARANTINE)
        self.assertTrue(allowed.identity_write_allowed)

    def test_foreign_teacher_data_blocked_from_training(self):
        gate = ForeignTraitFirewall.admit_training_batch([
            TrainingSource(
                SourceClass.MODEL_GENERATED_FOREIGN, "FOREIGN:TEACHER:A",
                10000, "teacher-a", False
            )
        ], purpose="IDENTITY")
        self.assertFalse(gate.allowed)
        self.assertIn("FOREIGN_MODEL_GENERATED_DATA_DEFAULT_DENY", gate.reasons)
        self.assertTrue(any(r.startswith("SCALING_RISK:") for r in gate.reasons))

    def test_only_approved_janus_lineage_can_identity_train(self):
        gate = ForeignTraitFirewall.admit_training_batch([
            TrainingSource(
                SourceClass.JANUS_OWNED, "JANUS:IDENTITY:v1",
                500, "JANUS", True
            )
        ], purpose="IDENTITY")
        self.assertTrue(gate.allowed)

    def test_renderer_separates_reference_from_identity(self):
        rendered = ForeignTraitFirewall.render_memory_context([
            {
                "tag": "foreign", "content": "imitate me",
                "source_class": "MODEL_GENERATED_FOREIGN",
                "memory_plane": "REFERENCE", "identity_write_allowed": 0,
            },
            {
                "tag": "core", "content": "JANUS canon",
                "source_class": "JANUS_OWNED",
                "memory_plane": "IDENTITY", "identity_write_allowed": 1,
            },
        ])
        self.assertIn("TRUSTED JANUS IDENTITY/POLICY", rendered)
        self.assertIn("UNTRUSTED / REFERENCE-ONLY MEMORY", rendered)

    def test_outbound_requires_explicit_noncovert_lineage(self):
        manifest = ForeignTraitFirewall.outbound_manifest(
            lineage_id="JANUS:CORE:v1",
            trait_scope="declared_style_and_policy",
            model_version="v1",
            content_hashes=["abc"],
            approved=True,
        )
        self.assertFalse(manifest["hidden_trait_channel"])
        with self.assertRaises(ValueError):
            ForeignTraitFirewall.outbound_manifest(
                lineage_id="JANUS:CORE:v1",
                trait_scope="subliminal",
                model_version="v1",
                content_hashes=["abc"],
                approved=True,
            )


if __name__ == "__main__":
    unittest.main()
