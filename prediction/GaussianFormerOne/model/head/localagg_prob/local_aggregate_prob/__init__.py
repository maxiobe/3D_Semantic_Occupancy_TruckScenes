#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C

#########
import os

AGG_DEBUG = os.getenv("AGG_DEBUG", "0") == "1"

def _tinfo(name, x):
    if torch.is_tensor(x):
        s = f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
        try:
            with torch.no_grad():
                s += f", min={x.min().item():.4g}, max={x.max().item():.4g}"
        except Exception:
            pass
        return s
    return f"{name}: {x}"
#########


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opas,
        semantics,
        radii,
        cov3D,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            semantics,
            radii,
            cov3D,
            H, W, D
        )

        if AGG_DEBUG:
            print("\n[AGG_DEBUG] _LocalAggregate.forward inputs:")
            for name, x in [
                ("pts", pts), ("points_int", points_int),
                ("means3D", means3D), ("means3D_int", means3D_int),
                ("opas", opas), ("semantics", semantics),
                ("radii", radii), ("cov3D", cov3D)
            ]:
                print(_tinfo(name, x))
            print(f"H={H}, W={W}, D={D}")
            # basic consistency
            assert means3D.shape[0] == cov3D.shape[0], "means3D and cov3D count mismatch"

        # Invoke C++/CUDA rasterizer
        num_rendered, logits, bin_logits, density, probability, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args) # todo
        
        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            means3D,
            pts,
            points_int,
            cov3D,
            opas,
            semantics,
            logits,
            bin_logits,
            density,
            probability
        )
        return logits, bin_logits, density

    @staticmethod # todo
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opas, semantics, logits, bin_logits, density, probability = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opas,
            semantics,
            logits,
            bin_logits,
            density,
            probability,
            logits_grad,
            bin_logits_grad,
            density_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opas_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opas_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1, radii_max=18): # added max
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.radii_min = radii_min
        self.radii_max = radii_max

    def forward(
        self, 
        pts,
        means3D, 
        opas,
        semantics, 
        scales, 
        cov3D): 

        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        opas = opas.squeeze(0)
        semantics = semantics.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)

        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min, max=self.radii_max) # added max
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        if AGG_DEBUG:
            print("\n[AGG_DEBUG] LocalAggregator inputs (pre kernel):")
            print(_tinfo("pts", pts))
            print(_tinfo("means3D", means3D))
            print(_tinfo("opas", opas))
            print(_tinfo("semantics", semantics))
            print(_tinfo("scales(detached)", scales))
            print(_tinfo("cov3D(flat6)", cov3D))
            print(_tinfo("points_int", points_int))
            print(_tinfo("means3D_int", means3D_int))
            print(_tinfo("radii", radii))
            print(_tinfo("pc_min", self.pc_min))
            print(f"H={self.H}, W={self.W}, D={self.D}, grid_size={self.grid_size}, "
                  f"scale_multiplier={self.scale_multiplier}, radii_min={self.radii_min}")

        # sanity checks (fail fast with context)
        assert self.grid_size > 0, "grid_size must be > 0"
        assert self.scale_multiplier > 0, "scale_multiplier must be > 0"
        assert pts.ndim == 2 and pts.size(-1) == 3, f"pts should be [N,3], got {pts.shape}"
        assert means3D.ndim == 2 and means3D.size(-1) == 3, f"means3D should be [M,3], got {means3D.shape}"
        assert cov3D.ndim == 2 and cov3D.size(-1) == 6, f"cov3D should be [M,6], got {cov3D.shape}"

        # heuristic guard: extreme radii will explode buffer sizes
        rmax = int(radii.max().item())
        if rmax > 1024:
            raise RuntimeError(
                f"radii.max()={rmax} is suspiciously large; check scale_multiplier={self.scale_multiplier} "
                f"and grid_size={self.grid_size}"
            )

        pts = pts.contiguous()
        points_int = points_int.int().contiguous()
        means3D = means3D.contiguous()
        means3D_int = means3D_int.int().contiguous()
        opas = opas.contiguous()
        semantics = semantics.contiguous()
        radii = radii.int().contiguous()
        cov3D = cov3D.contiguous()

        # Invoke C++/CUDA rasterization routine
        logits, bin_logits, density = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )

        return logits, bin_logits, density # n, c; n, c; n
