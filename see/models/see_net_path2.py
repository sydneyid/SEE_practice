"""
SEENet Path 2: prompt-free decode with K-step event-conditioned exposure refinement.
No exposure prompt B; output is trained directly against paired normal-light GT (NL).
"""

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


class ExposureRefineStep(nn.Module):
    """One shared refinement step: RGB state + fused features + events (cross-attention)."""

    def __init__(self, C1, C2, sparse_encoder_config):
        super().__init__()
        self.rgb_to_feat = nn.Conv2d(3, C2, kernel_size=1, stride=1, padding=0, bias=False)
        self.ev_to_feat = nn.Conv2d(C1, C2, kernel_size=1, stride=1, padding=0, bias=False)
        cfg = sparse_encoder_config
        self.fuse = SwinTransformerDecoderBlock(
            dim=C2, depth=cfg.depth, heads=cfg.heads, windows_size=cfg.windows_size
        )
        self.feat_to_rgb = nn.Conv2d(C2, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.adaptive_gate = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, rgb, light_inr, ev_feat, use_adaptive_residual=True):
        feat = self.rgb_to_feat(rgb)
        guide = self.ev_to_feat(ev_feat)
        delta_feat = self.fuse(feat, light_inr)
        delta_feat = delta_feat + guide
        delta_rgb = self.feat_to_rgb(delta_feat)
        if use_adaptive_residual:
            gate = self.adaptive_gate(torch.cat([rgb, delta_rgb], dim=1))
            rgb = rgb + gate * delta_rgb
        else:
            rgb = rgb + delta_rgb
        return rgb


class SEENetPath2(nn.Module):
    """
    SEE-Net encoder (Bayer PE + cross-attention sparse fusion) +
    init RGB head + K iterative exposure refinement steps (no exposure prompt).
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
        self.refine_steps = int(getattr(SEE_config, "refine_steps", 2))
        self.use_adaptive_residual = bool(getattr(SEE_config, "use_adaptive_residual", True))
        self.supervise_intermediate = bool(getattr(SEE_config, "supervise_intermediate", False))

        if self.SEE_config.position_embedding == "bayer_pattern":
            pos_channels = 4 if self.SEE_config.w_xy_coords else 2
            self.position_embedding = PositionEmbedding(self.SEE_config.position_embedding_type, pos_channels, C1)

        self.image_head, self.event_head = self._build_event_image_heads()
        self.scn_1 = _SparseEncoder(C1, C2, loop, sparse_encoder_config=self.SEE_config.sparse_encoder_config)
        self.init_rgb = nn.Conv2d(C2, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.refine_step = ExposureRefineStep(C1, C2, self.SEE_config.sparse_encoder_config)
        info(f"SEENetPath2: refine_steps={self.refine_steps}, adaptive_residual={self.use_adaptive_residual}")
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

    def forward(self, batch):
        events = batch[ELB.E]
        images = batch[ELB.LL]
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

        rgb = self.init_rgb(light_inr)
        intermediates = [rgb]
        for _ in range(self.refine_steps):
            rgb = self.refine_step(
                rgb, light_inr, ev, use_adaptive_residual=self.use_adaptive_residual
            )
            intermediates.append(rgb)

        batch[ELB.PRD] = rgb
        if self.training and self.supervise_intermediate:
            batch["PRD_INTERMEDIATE"] = torch.stack(intermediates[1:], dim=1)
        return batch
