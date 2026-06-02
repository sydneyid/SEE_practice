"""
SEENet Path 2: prompt-free iterative exposure refinement (ExposureDiffusion-inspired).

Integrates key ideas from ExposureDiffusion (wyf0912/ExposureDiffusion):
  - concat_origin: always condition on the original low-light frame
  - adaptive_res_and_x0: blend direct prediction and residual path with a learned mask
  - exposure ratio schedule + step embedding per refine iteration
  - per-step supervision (optional adaptive step weights via loss config)
"""

import math

import torch
import torch.nn as nn
from absl.logging import info
from torch.nn import functional as F

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.models.see_net import (
    PositionEmbedding,
    SwinTransformerDecoderBlock,
    _SparseEncoder,
    get_bayer_pattern_coordinate,
)
from see.utils.model_size import model_size


class ConvRefineFuse(nn.Module):
    """Lightweight conv fusion of refine features with encoder output (no attention)."""

    def __init__(self, C2):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(C2 * 2, C2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C2, C2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat, light_inr):
        return self.fuse(torch.cat([feat, light_inr], dim=1))


def _exposure_ratio_schedule(num_steps, ratio_max, schedule="linspace", device=None):
    """ED-style discrete exposure ratios from 1 -> ratio_max (inclusive endpoints)."""
    if num_steps <= 0:
        return torch.tensor([1.0, float(ratio_max)], device=device)
    if schedule == "linspace":
        return torch.linspace(1.0, float(ratio_max), num_steps + 1, device=device)
    if schedule == "logspace":
        return torch.logspace(0.0, math.log10(float(ratio_max)), num_steps + 1, device=device)
    raise ValueError(f"Unknown exposure schedule: {schedule}")


class ExposureStepCondition(nn.Module):
    """Embed normalized step index and log exposure ratio (ED iter conditioning)."""

    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, step_index, num_steps, log_ratio):
        if not torch.is_tensor(log_ratio):
            log_ratio = torch.tensor([math.log(max(float(log_ratio), 1.0))], dtype=torch.float32)
        if log_ratio.dim() == 0:
            log_ratio = log_ratio.unsqueeze(0)
        t_val = 0.0 if num_steps <= 1 else float(step_index) / float(num_steps - 1)
        t = torch.full((log_ratio.shape[0],), t_val, device=log_ratio.device, dtype=log_ratio.dtype)
        feat = torch.stack([t, log_ratio], dim=-1)
        return self.mlp(feat).unsqueeze(-1).unsqueeze(-1)


class ExposureRefineStep(nn.Module):
    """
    One refinement step: current RGB + optional LL anchor + step embed,
    fused with SEE sparse features and a precomputed event guide.
    """

    def __init__(
        self,
        C2,
        refine_backbone="conv",
        refine_encoder_config=None,
        concat_origin=True,
        adaptive_res_and_x0=True,
    ):
        super().__init__()
        self.concat_origin = concat_origin
        self.adaptive_res_and_x0 = adaptive_res_and_x0
        self.refine_backbone = refine_backbone
        rgb_in = 6 if concat_origin else 3
        self.rgb_to_feat = nn.Conv2d(rgb_in, C2, kernel_size=1, stride=1, padding=0, bias=False)
        self.step_to_feat = nn.Conv2d(C2, C2, kernel_size=1, stride=1, padding=0, bias=False)
        if refine_backbone == "conv":
            self.fuse = ConvRefineFuse(C2)
        elif refine_backbone == "swin":
            cfg = refine_encoder_config
            self.fuse = SwinTransformerDecoderBlock(
                dim=C2, depth=cfg.depth, heads=cfg.heads, windows_size=cfg.windows_size
            )
        else:
            raise ValueError(f"Unknown refine_backbone: {refine_backbone}")
        out_ch = 7 if adaptive_res_and_x0 else 3
        self.feat_to_rgb = nn.Conv2d(C2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        if not adaptive_res_and_x0:
            self.adaptive_gate = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, rgb, ll_origin, light_inr, event_guide, step_feat, use_adaptive_residual=True):
        x_in = torch.cat([rgb, ll_origin], dim=1) if self.concat_origin else rgb
        feat = self.rgb_to_feat(x_in)
        feat = feat + self.step_to_feat(step_feat)
        delta_feat = self.fuse(feat, light_inr)
        delta_feat = delta_feat + event_guide
        out = self.feat_to_rgb(delta_feat)
        metadata = None

        if self.adaptive_res_and_x0:
            mask = torch.sigmoid(out[:, [0]])
            x0 = out[:, 1:4].clamp(0.0, 1.0)
            resid = out[:, 4:7]
            out_by_resid = (resid + rgb).clamp(0.0, 1.0)
            rgb_next = x0 * mask + out_by_resid * (1.0 - mask)
            metadata = {"out": x0, "out_by_resid": out_by_resid, "mask": mask}
        else:
            delta_rgb = out
            if use_adaptive_residual:
                gate = self.adaptive_gate(torch.cat([rgb, delta_rgb], dim=1))
                rgb_next = rgb + gate * delta_rgb
            else:
                rgb_next = rgb + delta_rgb
        return rgb_next, metadata


class SEENetPath2(nn.Module):
    """
    SEE encoder + K-step ExposureDiffusion-style refinement (no exposure prompt B).
    """

    def __init__(self, frames, moments, C1, C2, loop, SEE_config):
        super().__init__()
        assert SEE_config.position_embedding in ["bayer_pattern", "none"]

        self.frames = frames
        self.in_channel = 3 * frames
        self.moments = moments
        self.C1 = C1
        self.C2 = C2
        self.loop = loop
        self.SEE_config = SEE_config

        self.refine_steps = int(getattr(SEE_config, "refine_steps", 3))
        self.concat_origin = bool(getattr(SEE_config, "concat_origin", True))
        self.adaptive_res_and_x0 = bool(getattr(SEE_config, "adaptive_res_and_x0", True))
        self.use_adaptive_residual = bool(getattr(SEE_config, "use_adaptive_residual", True))
        self.start_from_ll = bool(getattr(SEE_config, "start_from_ll", True))
        self.use_init_head = bool(getattr(SEE_config, "use_init_head", True))
        self.exposure_state_update = bool(getattr(SEE_config, "exposure_state_update", False))
        self.supervise_intermediate = bool(getattr(SEE_config, "supervise_intermediate", True))
        self.exposure_ratio_max = float(getattr(SEE_config, "exposure_ratio_max", 100.0))
        self.exposure_schedule = str(getattr(SEE_config, "exposure_schedule", "linspace"))
        self.use_batch_exposure_ratio = bool(getattr(SEE_config, "use_batch_exposure_ratio", True))
        self.refine_backbone = str(getattr(SEE_config, "refine_backbone", "conv"))
        refine_enc = getattr(SEE_config, "refine_encoder_config", None)
        self.refine_encoder_config = refine_enc if refine_enc is not None else SEE_config.sparse_encoder_config

        if self.SEE_config.position_embedding == "bayer_pattern":
            pos_channels = 4 if self.SEE_config.w_xy_coords else 2
            self.position_embedding = PositionEmbedding(self.SEE_config.position_embedding_type, pos_channels, C1)

        self.image_head, self.event_head = self._build_event_image_heads()
        self.scn_1 = _SparseEncoder(C1, C2, loop, sparse_encoder_config=self.SEE_config.sparse_encoder_config)
        self.step_condition = ExposureStepCondition(C2)
        if self.use_init_head:
            self.init_rgb = nn.Conv2d(C2, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.event_guide_proj = nn.Conv2d(C1, C2, kernel_size=1, stride=1, padding=0, bias=False)
        self.refine_step = ExposureRefineStep(
            C2,
            refine_backbone=self.refine_backbone,
            refine_encoder_config=self.refine_encoder_config,
            concat_origin=self.concat_origin,
            adaptive_res_and_x0=self.adaptive_res_and_x0,
        )
        enc_type = self.SEE_config.sparse_encoder_config.type
        info(
            f"SEENetPath2: steps={self.refine_steps}, encoder={enc_type}, refine_backbone={self.refine_backbone}, "
            f"concat_origin={self.concat_origin}, adaptive_res_and_x0={self.adaptive_res_and_x0}"
        )
        info(f"SEENetPath2 refine_step size: {model_size(self.refine_step)}")

    def _build_event_image_heads(self):
        position_embedding = self.C1 if self.SEE_config.position_embedding == "bayer_pattern" else 0
        image_in_channel = self.in_channel + position_embedding
        event_in_channel = self.moments + position_embedding
        if self.SEE_config.head == "original:image1x1-event1x1sigmod1x1":
            image_head = nn.Conv2d(
                in_channels=image_in_channel,
                out_channels=self.C1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            event_head = nn.Sequential(
                nn.Conv2d(
                    in_channels=event_in_channel,
                    out_channels=self.C1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Sigmoid(),
                nn.Conv2d(
                    in_channels=self.C1,
                    out_channels=self.C1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.Sigmoid(),
            )
            return image_head, event_head
        if self.SEE_config.head == "v1:w-9x9-depth-cpnv":
            image_head = nn.Sequential(
                nn.Conv2d(image_in_channel, self.C1, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.C1, self.C1, 9, 1, 4, bias=False, groups=self.C1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.C1, self.C1, 1, 1, 0, bias=False),
            )
            event_head = nn.Sequential(
                nn.Conv2d(event_in_channel, self.C1, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.C1, self.C1, 9, 1, 4, bias=False, groups=self.C1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.C1, self.C1, 1, 1, 0, bias=False),
            )
            return image_head, event_head
        raise ValueError(f"Unknown head: {self.SEE_config.head}")

    def _batch_exposure_max(self, ll, nl):
        ll_mean = ll.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
        nl_mean = nl.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
        ratio = (nl_mean / ll_mean).clamp(min=1.0, max=self.exposure_ratio_max)
        return ratio.squeeze(-1).squeeze(-1).squeeze(-1)

    def forward(self, batch):
        events = batch[ELB.E]
        images = batch[ELB.LL]
        ll_origin = images[:, :3]
        B, _, H, W = images.shape

        if self.SEE_config.position_embedding == "bayer_pattern":
            xy_pos = get_bayer_pattern_coordinate(h=H, w=W, w_xy_coords=self.SEE_config.w_xy_coords)
            xy_pos = xy_pos.unsqueeze(0).repeat(B, 1, 1, 1).to(images.device)
            xy_pos = self.position_embedding(xy_pos)
            images = torch.cat([images, xy_pos], dim=1)
            events = torch.cat([events, xy_pos], dim=1)

        x1 = self.image_head(images)
        ev = self.event_head(events)
        light_inr = self.scn_1(x1, ev)
        event_guide = self.event_guide_proj(ev)

        if self.start_from_ll:
            rgb = ll_origin.clone()
        else:
            rgb = torch.zeros_like(ll_origin)
        if self.use_init_head:
            rgb = (rgb + self.init_rgb(light_inr)).clamp(0.0, 1.0)

        if self.use_batch_exposure_ratio and ELB.NL in batch:
            ratio_max = self._batch_exposure_max(ll_origin, batch[ELB.NL])
        else:
            ratio_max = torch.full((B,), self.exposure_ratio_max, device=images.device, dtype=torch.float32)

        intermediates = []
        metadata_list = []
        device = images.device

        for step_i in range(self.refine_steps):
            r_curr = []
            r_next = []
            for b in range(B):
                sched = _exposure_ratio_schedule(
                    self.refine_steps, float(ratio_max[b].item()), self.exposure_schedule, device=device
                )
                r_curr.append(sched[step_i])
                r_next.append(sched[step_i + 1])
            r_curr_t = torch.tensor(r_curr, device=device, dtype=torch.float32)
            r_next_t = torch.tensor(r_next, device=device, dtype=torch.float32)
            log_r = torch.log(r_next_t.clamp(min=1.0))
            step_feat = self.step_condition(step_i, self.refine_steps, log_r)
            step_feat = step_feat.expand(B, self.C2, H, W)

            rgb, meta = self.refine_step(
                rgb,
                ll_origin,
                light_inr,
                event_guide,
                step_feat,
                use_adaptive_residual=self.use_adaptive_residual,
            )
            intermediates.append(rgb)
            if meta is not None:
                metadata_list.append(meta)

            if self.exposure_state_update and step_i + 1 < self.refine_steps:
                alpha = ((r_next_t - r_curr_t) / ratio_max.clamp(min=1.0)).view(B, 1, 1, 1)
                rgb = (ll_origin * (1.0 - alpha) + rgb * alpha).clamp(0.0, 1.0)

        batch[ELB.PRD] = rgb
        if self.training and self.supervise_intermediate and intermediates:
            batch["PRD_INTERMEDIATE"] = torch.stack(intermediates, dim=1)
        if metadata_list and self.adaptive_res_and_x0:
            batch["PRD_REFINE_METADATA"] = metadata_list
        return batch
