"""Toy-only V0.2 table checks; root runs these under its shared CPU supervisor."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "v02"))

from tables import ATOL, CONDITIONS, RTOL, build_tables, describe_tables, tables_check  # noqa: E402
from vocabpriorbench.model import next_token_labels, state_sha256, tensor_sha256  # noqa: E402
from vocabpriorbench.v01_model import OutputStudent, common_state_dict, load_common_state_dict  # noqa: E402


class RowNormDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(1729)
        self.C = torch.zeros(7, 128, dtype=torch.float32)
        self.C[0, :2] = torch.tensor([3.0, 4.0])
        self.C[1, :3] = torch.tensor([-2.0, 0.0, 2.0])
        self.C[3, 1] = -0.0
        self.C[3, 2] = 1e-20
        self.C[4] = 1.0
        self.C[5] = torch.linspace(-2.0, 3.0, 128)
        self.tables, self.r = build_tables(self.C)
        self.code_in = torch.randn(7, 128) / math.sqrt(128)
        self.counts = torch.tensor([1, 3, 2, 0, 4, 2, 0])
        self.pi = (self.counts.double() + 0.1) / (int(self.counts.sum()) + 0.7)
        self.bias = self.pi.log().float()
        self.kwargs = dict(dim=8, layers=2, heads=2, ff_dim=24, max_seq_len=8, dropout=0.0)
        self.ids = torch.tensor([[0, 1, 3, 4, 5], [4, 2, 1, 0, 0]])
        self.mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        reference = OutputStudent(self.code_in, self.tables["C"], False, self.bias, **self.kwargs)
        self.common = common_state_dict(reference)

    def model(self, name: str) -> OutputStudent:
        # Deliberately explicit: no condition-name prefix/suffix dispatch.
        model = OutputStudent(self.code_in, self.tables[name], False, self.bias, **self.kwargs)
        load_common_state_dict(model, self.common)
        return model

    def test_exact_anchors_zero_tiny_and_nonunit_rows(self) -> None:
        self.assertEqual(tuple(self.tables), CONDITIONS)
        self.assertEqual(tensor_sha256(self.tables["C"]), tensor_sha256(self.C))
        sign = torch.where(self.C >= 0, 1.0, -1.0) / math.sqrt(128)
        self.assertEqual(tensor_sha256(self.tables["S"]), tensor_sha256(sign))
        self.assertEqual(self.r.dtype, torch.float64)
        self.assertEqual(float(self.r[0]), 5.0)
        self.assertTrue(torch.equal(self.tables["U"][2], self.tables["S"][2]))
        self.assertTrue(torch.equal(self.tables["RS"][2], torch.zeros(128)))
        self.assertGreater(float(self.tables["S"][3, 1]), 0, "sign(-0) must be +1")
        self.assertEqual(float(self.tables["U"][3, 2]), 1.0, "positive tiny rows must not be epsilon-clamped")
        self.assertTrue(torch.isfinite(self.tables["U"]).all())
        checks = tables_check(self.tables, self.r, source_C=self.C)
        self.assertTrue(checks["passed"], checks)
        self.assertEqual((checks["rtol"], checks["atol"]), (2e-6, 2e-7))
        self.assertTrue(all(item["max_abs_error"] >= 0 for item in checks["actual_errors"].values()))
        self.assertEqual(len({table.data_ptr() for table in self.tables.values()}), 4)
        self.assertNotEqual(self.tables["C"].data_ptr(), self.C.data_ptr())
        with self.assertRaises(ValueError):
            build_tables(self.C.double())

    def test_description_reports_actual_hashes_norms_and_metadata(self) -> None:
        description = describe_tables(
            self.tables, self.r, source_C=self.C, pi=self.pi, counts=self.counts,
            special_token_ids={"eos": 2, "pad": None}, tokenizer_length=6,
        )
        self.assertTrue(description["checks"]["passed"])
        self.assertEqual(description["zero_row_ids"], [2, 6])
        self.assertEqual(description["positive_tiny_row_ids"], [3])
        self.assertEqual(description["zero_row_metadata"][0]["training_target_count"], 2)
        self.assertEqual(description["zero_row_metadata"][0]["special_token_names"], ["eos"])
        self.assertTrue(description["zero_row_metadata"][1]["tokenizer_unused_output_row"])
        self.assertEqual(description["training_target_count_total"], int(self.counts.sum()))
        for name in CONDITIONS:
            item = description["tables"][name]
            self.assertEqual(item["tensor_sha256"], tensor_sha256(self.tables[name]))
            self.assertEqual(set(item["row_norm_quantiles"]), {"min", "p1", "p25", "median", "p75", "p99", "max"})
            expected_energy = float((self.pi.dot(self.tables[name].double().square().sum(1)) / 128).sqrt())
            self.assertAlmostEqual(item["training_pi_weighted_RMS"], expected_energy, places=14)

    def test_explicit_output_scores_then_unscaled_bias_and_paired_states(self) -> None:
        models = {name: self.model(name).eval() for name in CONDITIONS}
        expected_common = state_sha256(self.common)
        for name, model in models.items():
            self.assertEqual(tensor_sha256(model.output_codes), tensor_sha256(self.tables[name]))
            self.assertEqual(tensor_sha256(model.input_codes), tensor_sha256(self.code_in))
            self.assertEqual(tensor_sha256(model.output_bias), tensor_sha256(self.bias))
            self.assertEqual(state_sha256(common_state_dict(model)), expected_common)
            captured = {}
            hook = model.output_projection.register_forward_hook(lambda module, inputs, output: captured.update(q=output.detach()))
            with torch.no_grad():
                actual = model(self.ids, self.mask)
            hook.remove()
            expected = F.linear(captured["q"].float(), self.tables[name]) + self.bias
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
            if name == "RS":
                # Zero RS rows must output the unchanged log prior, not zero.
                torch.testing.assert_close(actual[..., 2], self.bias[2].expand_as(actual[..., 2]), rtol=0, atol=0)
                wrong = (F.linear(captured["q"].float(), self.tables["S"]) + self.bias) * self.r.float()
                self.assertGreater(float((actual - wrong).abs().max()), 0.1)
        for attr in ["input_codes", "output_codes", "output_bias"]:
            self.assertEqual(len({getattr(model, attr).data_ptr() for model in models.values()}), 4)

    def test_all_four_frozen_interventions_and_public_updates(self) -> None:
        labels = next_token_labels(self.ids, self.mask)
        self.assertEqual(int((labels != -100).sum()), 6)
        source_hashes = {name: tensor_sha256(table) for name, table in self.tables.items()}
        for name in CONDITIONS:
            with self.subTest(condition=name):
                model = self.model(name)
                fixed_names = ["input_codes", "output_codes", "output_bias"]
                before_fixed = {key: tensor_sha256(getattr(model, key)) for key in fixed_names}
                before_common = common_state_dict(model)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
                loss = F.cross_entropy(model(self.ids, self.mask).flatten(0, 1), labels.flatten(), ignore_index=-100)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter_name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, parameter_name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                        self.assertGreater(float(parameter.grad.abs().max()), 0, parameter_name)
                    else:
                        self.assertIsNone(parameter.grad, parameter_name)
                self.assertIsNone(model.output_bias.grad)
                optimizer.step()
                after_common = common_state_dict(model)
                for key in before_common:
                    self.assertTrue(torch.isfinite(after_common[key]).all(), key)
                    self.assertFalse(torch.equal(before_common[key], after_common[key]), key)
                self.assertEqual(before_fixed, {key: tensor_sha256(getattr(model, key)) for key in fixed_names})
        self.assertEqual(source_hashes, {name: tensor_sha256(table) for name, table in self.tables.items()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
