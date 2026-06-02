"""Per-step supervision for iterative refinement models (ExposureDiffusion-style)."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.losses.image_loss import L1CharbonnierLoss


class IterativeRefinementLoss(_Loss):
    """
    Supervise each refine step against normal-light GT.
    Optional learnable step weights (adaptive_loss) and aux branch loss (aux_loss).
    """

    def __init__(self, configs):
        super().__init__()
        self.pixel_loss = L1CharbonnierLoss()
        self.adaptive_loss = bool(getattr(configs, "adaptive_loss", False))
        self.aux_loss = bool(getattr(configs, "aux_loss", False))
        self.aux_weight = float(getattr(configs, "aux_weight", 0.25))
        num_steps = int(getattr(configs, "refine_steps", 3))
        if self.adaptive_loss:
            init = torch.ones(num_steps) / (num_steps + 1)
            self.step_weights = nn.Parameter(init)
        else:
            schedule = str(getattr(configs, "step_weight_schedule", "increasing"))
            self.register_buffer("fixed_weights", self._fixed_weights(num_steps, schedule))

    @staticmethod
    def _fixed_weights(num_steps, schedule):
        if num_steps <= 0:
            return torch.ones(1)
        if schedule == "uniform":
            return torch.ones(num_steps) / num_steps
        if schedule == "increasing":
            w = torch.linspace(1.0, float(num_steps), num_steps)
            return w / w.sum()
        if schedule == "last_only":
            w = torch.zeros(num_steps)
            w[-1] = 1.0
            return w
        raise ValueError(f"Unknown step_weight_schedule: {schedule}")

    def _weight(self, step_idx, device):
        if self.adaptive_loss:
            w = self.step_weights[step_idx].clamp(0.0, 1.0)
            return w
        return self.fixed_weights[step_idx].to(device)

    def forward(self, batch):
        target = batch[ELB.NL]
        if "PRD_INTERMEDIATE" not in batch:
            return self.pixel_loss(target, batch[ELB.PRD])

        preds = batch["PRD_INTERMEDIATE"]
        K = preds.shape[1]
        metadata = batch.get("PRD_REFINE_METADATA", None)
        total = preds.new_tensor(0.0)
        device = preds.device

        for k in range(K):
            step_loss = self.pixel_loss(target, preds[:, k])
            if self.aux_loss and metadata is not None and k < len(metadata):
                meta = metadata[k]
                mask = meta["mask"]
                aux = (torch.abs(meta["out"] - target) * mask).mean()
                aux = aux + (torch.abs(meta["out_by_resid"] - target) * (1.0 - mask)).mean()
                step_loss = step_loss + self.aux_weight * aux
            total = total + step_loss * self._weight(k, device)

        if self.adaptive_loss:
            remainder = (1.0 - self.step_weights.clamp(0.0, 1.0).sum()).clamp(min=0.0)
            total = total + self.pixel_loss(target, batch[ELB.PRD]) * remainder
        return total
