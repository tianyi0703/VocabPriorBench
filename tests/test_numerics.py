"""Discriminating CPU checks; run `python -m unittest discover -s tests -v`."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vocabpriorbench.model import (  # noqa: E402
    Student,
    common_state_dict,
    load_common_state_dict,
    next_token_labels,
    state_sha256,
    tensor_sha256,
    unique_parameter_count,
)


class NumericalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(20261021)
        self.code_in = torch.randn(31, 8) / math.sqrt(8)
        self.code_out = torch.randn(31, 8) / math.sqrt(8)
        self.ids = torch.tensor([[2, 7, 4, 9, 3, 0], [6, 3, 8, 1, 0, 0]])
        self.mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])

    def student(self, trainable: bool = False, binary: bool = False) -> Student:
        code_in, code_out = self.code_in, self.code_out
        if binary:
            code_in = torch.where(code_in >= 0, 1.0, -1.0) / math.sqrt(8)
            code_out = torch.where(code_out >= 0, 1.0, -1.0) / math.sqrt(8)
        return Student(
            code_in,
            code_out,
            trainable,
            dim=8,
            layers=2,
            heads=2,
            ff_dim=24,
            max_seq_len=8,
            dropout=0.0,
        )

    def test_causality_future_does_not_change_past(self) -> None:
        model = self.student().eval()
        altered = self.ids.clone()
        altered[:, 3:] = torch.tensor([[30, 25, 12], [14, 21, 15]])
        with torch.no_grad():
            original = model(self.ids)
            changed = model(altered)
            prefix = model(self.ids[:, :3])
        torch.testing.assert_close(original[:, :3], changed[:, :3], rtol=0, atol=1e-6)
        torch.testing.assert_close(original[:, :3], prefix, rtol=0, atol=1e-6)
        self.assertGreater((original[:, 3:] - changed[:, 3:]).abs().max().item(), 1e-4)

    def test_attention_padding_masks_keys_and_remains_finite(self) -> None:
        model = self.student().eval()
        # An interior hole tests key masking independently of causal masking.
        ids = torch.tensor([[3, 5, 6, 8, 9]])
        mask = torch.tensor([[1, 0, 1, 1, 1]])
        changed_ids = ids.clone()
        changed_ids[0, 1] = 25
        with torch.no_grad():
            actual = model(ids, mask)
            changed = model(changed_ids, mask)
            unmasked = model(ids)
            all_padding = model(ids, torch.zeros_like(mask))
        torch.testing.assert_close(actual[:, 2:], changed[:, 2:], rtol=0, atol=1e-6)
        self.assertGreater((actual[:, 2:] - unmasked[:, 2:]).abs().max().item(), 1e-5)
        self.assertTrue(torch.isfinite(all_padding).all())

    def test_next_token_shift_padding_and_token_weighted_loss(self) -> None:
        labels = next_token_labels(self.ids, self.mask)
        expected = torch.tensor([[7, 4, 9, 3, -100, -100], [3, 8, 1, -100, -100, -100]])
        self.assertTrue(torch.equal(labels, expected))
        logits = torch.randn(2, 6, 31, requires_grad=True)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        reference_terms = []
        for row, length in [(0, 5), (1, 4)]:
            for position in range(length - 1):
                reference_terms.append(-logits[row, position].log_softmax(-1)[self.ids[row, position + 1]])
        reference = torch.stack(reference_terms).sum() / 7
        torch.testing.assert_close(loss, reference)
        grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        reference_grad = torch.autograd.grad(reference, logits)[0]
        torch.testing.assert_close(grad, reference_grad)
        self.assertTrue(torch.equal(grad[labels == -100], torch.zeros_like(grad[labels == -100])))
        left_ids = torch.tensor([[0, 0, 3, 4, 5]])
        left_mask = torch.tensor([[0, 0, 1, 1, 1]])
        self.assertTrue(torch.equal(next_token_labels(left_ids, left_mask), torch.tensor([[-100, -100, 4, 5, -100]])))

    def test_a_b_saved_common_state_and_forward_identical(self) -> None:
        a = self.student(trainable=True).eval()
        initial = common_state_dict(a)
        initial_hash = state_sha256(initial)
        b = self.student(trainable=False).eval()
        self.assertNotEqual(state_sha256(common_state_dict(b)), initial_hash)
        load_common_state_dict(b, initial)
        self.assertEqual(state_sha256(common_state_dict(a)), initial_hash)
        self.assertEqual(state_sha256(common_state_dict(b)), initial_hash)
        self.assertTrue(torch.equal(a.input_codes, b.input_codes))
        self.assertTrue(torch.equal(a.output_codes, b.output_codes))
        with torch.no_grad():
            torch.testing.assert_close(a(self.ids, self.mask), b(self.ids, self.mask), rtol=0, atol=0)
        broken = dict(initial)
        broken.pop(next(iter(broken)))
        with self.assertRaises(ValueError):
            load_common_state_dict(b, broken)

    def test_all_arms_common_parameters_update_and_codes_obey_freeze(self) -> None:
        initial = common_state_dict(self.student())
        labels = next_token_labels(self.ids, self.mask)
        for arm, trainable, binary in [("A", True, False), ("B", False, False), ("C", False, True)]:
            with self.subTest(arm=arm):
                model = self.student(trainable=trainable, binary=binary)
                load_common_state_dict(model, initial)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.05)
                logits = model(self.ids, self.mask)
                loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, f"{arm}/{name} has no gradient")
                        self.assertTrue(torch.isfinite(parameter.grad).all(), f"{arm}/{name} non-finite gradient")
                        self.assertGreater(parameter.grad.abs().max().item(), 0.0, f"{arm}/{name} zero gradient")
                    else:
                        self.assertIsNone(parameter.grad, f"{arm}/{name} frozen gradient")
                optimizer.step()
                for name, parameter in model.named_parameters():
                    self.assertTrue(torch.isfinite(parameter).all(), f"{arm}/{name} non-finite parameter")
                    if parameter.requires_grad:
                        self.assertFalse(torch.equal(before[name], parameter), f"{arm}/{name} did not update")
                    else:
                        self.assertTrue(torch.equal(before[name], parameter), f"{arm}/{name} frozen code changed")

    def test_untied_code_identity_and_unique_parameter_counts(self) -> None:
        model = self.student(trainable=True)
        self.assertIsNot(model.input_codes, model.output_codes)
        self.assertNotEqual(model.input_codes.data_ptr(), model.output_codes.data_ptr())
        self.assertEqual(unique_parameter_count(model), sum(p.numel() for p in model.parameters()))
        public_count = sum(t.numel() for t in common_state_dict(model).values())
        self.assertEqual(unique_parameter_count(model), public_count + 2 * 31 * 8)
        frozen = self.student()
        self.assertEqual(unique_parameter_count(frozen, trainable_only=True), public_count)
        # Aliased registrations must not inflate counts.
        aliased = nn.Module()
        aliased.first = model.input_codes
        aliased.second = model.input_codes
        self.assertEqual(unique_parameter_count(aliased), 31 * 8)
        # Passing the same source Tensor must still construct two storage objects.
        same_source = Student(self.code_in, self.code_in, True, dim=8, heads=2, ff_dim=24)
        self.assertNotEqual(same_source.input_codes.data_ptr(), same_source.output_codes.data_ptr())
        self.assertIsNone(model.input_projection.bias)
        self.assertIsNone(model.output_projection.bias)
        self.assertNotIn("output_bias", model.state_dict())

    def test_hashes_are_stable_sensitive_and_do_not_alias(self) -> None:
        model = self.student()
        initial = common_state_dict(model)
        initial_hash = state_sha256(initial)
        self.assertEqual(initial_hash, state_sha256(dict(reversed(list(initial.items())))))
        with torch.no_grad():
            model.input_projection.weight[0, 0].add_(1)
        self.assertEqual(initial_hash, state_sha256(initial))
        self.assertNotEqual(initial_hash, state_sha256(common_state_dict(model)))
        value = torch.tensor(1.0)
        self.assertEqual(tensor_sha256(value), tensor_sha256(value.clone()))
        self.assertNotEqual(tensor_sha256(value), tensor_sha256(value.reshape(1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
