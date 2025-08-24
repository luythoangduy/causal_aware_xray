from __future__ import annotations
import torch
from torch import nn, Tensor
from typing import List, Tuple, Optional


class ConstraintProjection(nn.Module):
    """Differentiable projection enforcing implication & exclusion constraints.

    Not a full Dykstra (no dual vars) but efficient cyclic projections.
    """

    def __init__(
        self,
        implications: Optional[List[Tuple[int, int, float]]] = None,
        exclusions: Optional[List[Tuple[int, int, float]]] = None,
        max_iter: int = 20,
        eps: float = 1e-4,
        track_history: bool = False,
    ):
        super().__init__()
        self.implications = implications or []
        self.exclusions = exclusions or []
        self.max_iter = max_iter
        self.eps = eps
        self.track_history = track_history

        if self.implications:
            self.register_buffer('imp_i', torch.tensor([i for i, _, _ in self.implications], dtype=torch.long))
            self.register_buffer('imp_j', torch.tensor([j for _, j, _ in self.implications], dtype=torch.long))
            self.register_buffer('imp_tau', torch.tensor([t for _, _, t in self.implications], dtype=torch.float32))
        else:
            self.register_buffer('imp_i', torch.empty(0, dtype=torch.long))
            self.register_buffer('imp_j', torch.empty(0, dtype=torch.long))
            self.register_buffer('imp_tau', torch.empty(0, dtype=torch.float32))

        if self.exclusions:
            self.register_buffer('exc_i', torch.tensor([i for i, _, _ in self.exclusions], dtype=torch.long))
            self.register_buffer('exc_j', torch.tensor([j for _, j, _ in self.exclusions], dtype=torch.long))
            self.register_buffer('exc_kappa', torch.tensor([k for _, _, k in self.exclusions], dtype=torch.float32))
        else:
            self.register_buffer('exc_i', torch.empty(0, dtype=torch.long))
            self.register_buffer('exc_j', torch.empty(0, dtype=torch.long))
            self.register_buffer('exc_kappa', torch.empty(0, dtype=torch.float32))

    def forward(self, logits: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        if (self.imp_i.numel() + self.exc_i.numel()) == 0:
            return probs
        prev = probs
        for _ in range(self.max_iter):
            if self.imp_i.numel():
                qi = probs[:, self.imp_i.to(probs.device)]
                qj = probs[:, self.imp_j.to(probs.device)]
                tau = self.imp_tau.to(probs.device).view(1, -1)
                violation = (qi + tau) - qj
                if violation.any():
                    adj = torch.clamp(violation, min=0)
                    qj_new = torch.clamp(qj + adj, 0.0, 1.0)
                    probs = probs.scatter(1, self.imp_j.to(probs.device).expand(probs.size(0), -1), qj_new)
            if self.exc_i.numel():
                qi = probs[:, self.exc_i.to(probs.device)]
                qj = probs[:, self.exc_j.to(probs.device)]
                kappa = self.exc_kappa.to(probs.device).view(1, -1)
                sum_ij = qi + qj
                violation = sum_ij - kappa
                if violation.any():
                    over = torch.clamp(violation, min=0)
                    red = over / 2.0
                    qi_new = torch.clamp(qi - red, 0.0, 1.0)
                    qj_new = torch.clamp(qj - red, 0.0, 1.0)
                    probs = probs.scatter(1, self.exc_i.to(probs.device).expand(probs.size(0), -1), qi_new)
                    probs = probs.scatter(1, self.exc_j.to(probs.device).expand(probs.size(0), -1), qj_new)
            if (probs - prev).abs().max() < self.eps:
                break
            prev = probs
        return probs

    def constraint_violations(self, probs: Tensor) -> dict:
        out = {}
        if self.imp_i.numel():
            qi = probs[:, self.imp_i.to(probs.device)]
            qj = probs[:, self.imp_j.to(probs.device)]
            tau = self.imp_tau.to(probs.device).view(1, -1)
            imp_v = torch.clamp((qi + tau) - qj, min=0)
            out['implication_total'] = imp_v.sum().item()
            out['implication_mean'] = imp_v.mean().item()
        if self.exc_i.numel():
            qi = probs[:, self.exc_i.to(probs.device)]
            qj = probs[:, self.exc_j.to(probs.device)]
            kappa = self.exc_kappa.to(probs.device).view(1, -1)
            exc_v = torch.clamp((qi + qj) - kappa, min=0)
            out['exclusion_total'] = exc_v.sum().item()
            out['exclusion_mean'] = exc_v.mean().item()
        return out
