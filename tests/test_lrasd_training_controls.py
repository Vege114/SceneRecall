"""CPU-only adaptation-control checks, runnable with unittest in the LR-ASD venv."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


script = Path(__file__).resolve().parents[1] / "scripts" / "train_lrasd.py"
spec = importlib.util.spec_from_file_location("lrasd_training_controls_test", script)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TrainingControlConfigurationTests(unittest.TestCase):
    def test_default_controls_preserve_legacy_hash_and_nondefaults_reject_resume(self):
        args = trainer.parse_args(["--manifest", "unused.jsonl", "--output", "unused"])
        self.assertEqual((args.batchnorm_mode, args.trainable_scope), ("update", "all"))
        self.assertEqual(trainer.training_control_config(args), {})
        legacy = {"manifest_sha256": "fixed-manifest", "seed": 114}
        old_hash = trainer.canonical_hash(legacy)
        self.assertEqual(trainer.canonical_hash({**legacy, **trainer.training_control_config(args)}), old_hash)
        state = {"schema_version": "scenerecall-lrasd-resume-v1", "config": legacy, "config_hash": old_hash,
                 "model": {f"parameter-{i}": 0 for i in range(168)}, "optimizer": {}, "scheduler": {}, "rng": {},
                 "epoch": 0, "epochs_completed": 0, "batch_offset": 0, "steps": 0,
                 "best_epoch": 0, "best_ap": -1.0, "stale_epochs": 0, "baseline_validation": None}
        self.assertIs(trainer.validate_resume_checkpoint(state, old_hash), state)
        for bn, scope in (("frozen", "all"), ("update", "heads"), ("frozen", "heads")):
            controls = trainer.training_control_config(SimpleNamespace(batchnorm_mode=bn, trainable_scope=scope))
            current = {**legacy, **controls}
            with self.subTest(controls=controls):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    trainer.validate_resume_checkpoint(state, trainer.canonical_hash(current))
                current_state = {**state, "config": current, "config_hash": trainer.canonical_hash(current)}
                self.assertIs(trainer.validate_resume_checkpoint(current_state, current_state["config_hash"]),
                              current_state)
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    trainer.validate_resume_checkpoint(current_state, old_hash)


@unittest.skipIf(torch is None, "Run in the LR-ASD venv for the CPU torch checks")
class TrainingControlTorchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(114)
        torch.set_num_threads(1)

        class SmallASD(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Sequential(
                    torch.nn.BatchNorm3d(4),
                    torch.nn.AdaptiveAvgPool3d((1, 2, 2)),
                    torch.nn.Flatten(2, 3),
                    torch.nn.BatchNorm2d(4),
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(4, 8),
                    torch.nn.BatchNorm1d(8),
                    torch.nn.Tanh(),
                    torch.nn.Dropout(.2))
                self.lossAV = torch.nn.Module()
                self.lossAV.FC = torch.nn.Linear(8, 2)
                self.lossV = torch.nn.Module()
                self.lossV.FC = torch.nn.Linear(8, 2)

            def forward(self, inputs):
                encoded = self.model(inputs)
                return self.lossAV.FC(encoded), self.lossV.FC(encoded)

        self.model = SmallASD()
        self.inputs = torch.randn(6, 4, 3, 2, 2) + 4
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1])

    def optimize(self, scope, mode):
        counts = trainer.configure_trainable_scope(self.model, scope)
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=.01)
        for _ in range(2):
            # Validation between epochs must not undo the frozen-BN policy.
            self.model.eval()
            trainer.set_training_mode(torch, self.model, mode)
            optimizer.zero_grad(set_to_none=True)
            av, visual = self.model(self.inputs)
            loss = (torch.nn.functional.cross_entropy(av, self.labels)
                    + .5 * torch.nn.functional.cross_entropy(visual, self.labels))
            loss.backward()
            optimizer.step()
        return counts

    def test_frozen_bn_retains_all_running_buffers_while_affine_parameters_train(self):
        before = {name: value.clone() for name, value in self.model.named_buffers()}
        counts = self.optimize("all", "frozen")
        self.assertEqual(counts["total"], counts["trainable"])
        self.assertEqual(counts["frozen"], 0)
        for name, value in self.model.named_buffers():
            self.assertTrue(torch.equal(value, before[name]), name)
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.assertFalse(module.training)
                self.assertTrue(module.weight.requires_grad)
                self.assertIsNotNone(module.weight.grad)
        self.assertTrue(self.model.model[-1].training)  # Only BN is put in eval mode.

    def test_head_scope_changes_both_heads_and_preserves_every_backbone_parameter_and_buffer(self):
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        counts = self.optimize("heads", "frozen")
        head_names = {name for name, _ in self.model.named_parameters()
                      if name.startswith(("lossAV.FC.", "lossV.FC."))}
        self.assertEqual(counts["trainable"], sum(p.numel() for n, p in self.model.named_parameters()
                                               if n in head_names))
        self.assertGreater(counts["frozen"], 0)
        for name, value in self.model.state_dict().items():
            if name in head_names:
                self.assertFalse(torch.equal(value, before[name]), name)
            else:
                self.assertTrue(torch.equal(value, before[name]), name)
        for name, parameter in self.model.named_parameters():
            self.assertEqual(parameter.requires_grad, name in head_names)
            self.assertEqual(parameter.grad is not None, name in head_names)
        restored = trainer.configure_trainable_scope(self.model, "all")
        self.assertEqual(restored["frozen"], 0)

    def test_default_mode_updates_bn_buffers(self):
        self.optimize("all", "update")
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                self.assertTrue(module.training)
                self.assertEqual(module.num_batches_tracked.item(), 2)
                self.assertGreater(module.running_mean.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
