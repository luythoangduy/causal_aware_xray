from __future__ import annotations
import torch
from src.models.constraint_projection import ConstraintProjection


def test_implication_projection():
    layer = ConstraintProjection(implications=[(0, 1, 0.0)], exclusions=[], max_iter=5)
    logits = torch.tensor([[0.0, -2.0]])  # probs ~ [0.5, 0.119]
    probs_proj = layer(logits)
    assert probs_proj[0, 1] >= probs_proj[0, 0] - 1e-5, "Implication not satisfied"


def test_exclusion_projection():
    layer = ConstraintProjection(exclusions=[(0, 1, 0.6)], max_iter=5)
    logits = torch.tensor([[4.0, 4.0]])  # probs ~ [0.982, 0.982] sum>1.6
    probs_proj = layer(logits)
    assert (probs_proj[0, 0] + probs_proj[0, 1]) <= 0.6 + 1e-5, "Exclusion not enforced"
