"""CPU checks for V0.1 interventions; retain existing V0 checks unchanged."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vocabpriorbench.model import (  # noqa: E402
    Student,
    common_state_dict as base_common_state_dict,
    next_token_labels,
    state_sha256,
)
from vocabpriorbench.v01_model import (  # noqa: E402
    OutputStudent,
    common_state_dict,
    load_common_state_dict,
)


class OutputInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(1729)
        self.code_in = torch.randn(31, 8) / math.sqrt(8)
        self.code_out = torch.randn(31, 8) / math.sqrt(8)
        self.sign_out = torch.where(self.code_out >= 0, 1.0, -1.0) / math.sqrt(8)
        self.kwargs = dict(dim=8, layers=2, heads=2, ff_dim=24, max_seq_len=8, dropout=0.0)
        self.ids = torch.tensor([[2, 7, 4, 9, 3, 0], [6, 3, 8, 1, 0, 0]])
        self.mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
        self.labels = next_token_labels(self.ids, self.mask)
        counts = torch.bincount(self.labels[self.labels != -100], minlength=31).double()
        self.pi = (counts + 0.1) / (counts.sum() + 0.1 * 31)
        self.logpi = self.pi.log().float()
        self.zero = torch.zeros(31)
        base = Student(self.code_in, self.code_out, False, **self.kwargs)
        self.common = base_common_state_dict(base)

    def make(self, condition: str) -> OutputStudent:
        model = OutputStudent(
            self.code_in,
            self.sign_out if condition.startswith("S") else self.code_out,
            trainable_output=condition.startswith("L"),
            output_bias=self.logpi if condition.endswith("p") else self.zero,
            **self.kwargs,
        )
        load_common_state_dict(model, self.common)
        return model

    def test_same_bias_l_f_initial_forward_and_common_hash(self) -> None:
        common_hash = state_sha256(self.common)
        models = {condition: self.make(condition).eval() for condition in ["L0", "F0", "S0", "Lp", "Fp", "Sp"]}
        for condition, model in models.items():
            self.assertEqual(state_sha256(common_state_dict(model)), common_hash, condition)
            self.assertTrue(torch.equal(model.input_codes, self.code_in), condition)
            self.assertFalse(model.input_codes.requires_grad)
            self.assertNotIn("output_bias", dict(model.named_parameters()))
            self.assertEqual(model.output_bias.dtype, torch.float32)
            self.assertFalse(model.output_bias.requires_grad)
        with torch.no_grad():
            outputs = {condition: model(self.ids, self.mask) for condition, model in models.items()}
        for suffix in ["0", "p"]:
            torch.testing.assert_close(outputs["L" + suffix], outputs["F" + suffix], rtol=0, atol=0)
        # Every condition follows one identical addition path and public core.
        for letter in ["L", "F", "S"]:
            torch.testing.assert_close(outputs[letter + "p"] - outputs[letter + "0"], self.logpi.expand_as(outputs[letter + "0"]), rtol=0, atol=5e-7)
        for first, second in [("L0", "F0"), ("L0", "Lp"), ("S0", "Sp")]:
            self.assertNotEqual(models[first].input_codes.data_ptr(), models[second].input_codes.data_ptr())
            self.assertNotEqual(models[first].output_codes.data_ptr(), models[second].output_codes.data_ptr())
            self.assertNotEqual(models[first].output_bias.data_ptr(), models[second].output_bias.data_ptr())

    def test_six_conditions_freeze_update_finite_and_valid_count(self) -> None:
        self.assertEqual(int((self.labels != -100).sum()), 7)
        for condition in ["L0", "F0", "S0", "Lp", "Fp", "Sp"]:
            with self.subTest(condition=condition):
                model = self.make(condition)
                before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=0.01)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    logits = model(self.ids, self.mask)
                    nll_sum = F.cross_entropy(logits.flatten(0, 1), self.labels.flatten(), ignore_index=-100, reduction="sum")
                self.assertEqual(logits.dtype, torch.float32)
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(nll_sum))
                (nll_sum / 7).backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(parameter.grad.abs().max().item(), 0, name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                optimizer.step()
                for name, parameter in model.named_parameters():
                    self.assertTrue(torch.isfinite(parameter).all(), name)
                    if parameter.requires_grad:
                        self.assertFalse(torch.equal(before[name], parameter), name)
                    else:
                        self.assertTrue(torch.equal(before[name], parameter), name)
                self.assertTrue(torch.equal(before["output_bias"], model.output_bias))
                self.assertIsNone(model.output_bias.grad)
                self.assertTrue(torch.equal(before["input_codes"], model.input_codes))
                self.assertEqual(model.output_codes.requires_grad, condition.startswith("L"))

    def test_common_loading_preserves_bias_and_rejects_intervention_keys(self) -> None:
        model = self.make("Sp")
        before = {key: model.state_dict()[key].clone() for key in ["input_codes", "output_codes", "output_bias"]}
        load_common_state_dict(model, self.common)
        for key, value in before.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        bad = dict(self.common)
        bad["output_bias"] = self.zero
        with self.assertRaises(ValueError):
            load_common_state_dict(model, bad)
        with self.assertRaises(ValueError):
            OutputStudent(self.code_in, self.code_out, False, torch.zeros(30), **self.kwargs)
        with self.assertRaises(ValueError):
            OutputStudent(self.code_in, self.code_out, False, torch.full((31,), float("nan")), **self.kwargs)
        # Detached buffer construction must not alias caller storage or retain grad.
        caller_bias = self.logpi.clone().requires_grad_(True)
        independent = OutputStudent(self.code_in, self.code_out, False, caller_bias, **self.kwargs)
        with torch.no_grad():
            caller_bias.add_(1)
        self.assertTrue(torch.equal(independent.output_bias, self.logpi))
        self.assertFalse(independent.output_bias.requires_grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
