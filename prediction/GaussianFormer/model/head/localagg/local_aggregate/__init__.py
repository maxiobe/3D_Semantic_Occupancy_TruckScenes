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


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opacities,
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
            opacities,
            semantics,
            radii,
            cov3D,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, logits, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args) # todo
        
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
            opacities,
            semantics
        )
        return logits

    @staticmethod # todo
    def backward(ctx, out_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opacities, semantics = ctx.saved_tensors

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
            opacities,
            semantics,
            out_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opacity_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opacity_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

"""class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, inv_softmax=False):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.inv_softmax = inv_softmax

    def forward(
        self, 
        pts,
        means3D, 
        opacities, 
        semantics, 
        scales, 
        cov3D): 

        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        opacities = opacities.squeeze(0)
        semantics = semantics.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)

        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        # Invoke C++/CUDA rasterization routine
        logits = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )

        if not self.inv_softmax:
            return logits # n, c
        else:
            assert False"""


class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, inv_softmax=False):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.inv_softmax = inv_softmax

    def forward(
            self,
            pts,
            means3D,
            opacities,
            semantics,
            scales,
            cov3D):

        # --- Start of new batch handling logic ---
        batch_size = pts.shape[0]
        output_logits = []

        for i in range(batch_size):
            # Slice all tensors to get the i-th sample for processing
            # Indexing with [i] removes the batch dimension, which is what the original code expects
            pts_i = pts[i]
            means3D_i = means3D[i]
            opacities_i = opacities[i]
            semantics_i = semantics[i]
            scales_i = scales.detach()[i]  # Detach is still needed
            cov3D_i = cov3D[i]

            assert not pts_i.requires_grad

            # --- This is the original logic, now inside the loop ---
            points_int_i = ((pts_i - self.pc_min) / self.grid_size).to(torch.int)
            assert points_int_i.min() >= 0 and points_int_i[:, 0].max() < self.H and points_int_i[
                :, 1].max() < self.W and points_int_i[:, 2].max() < self.D

            means3D_int_i = ((means3D_i.detach() - self.pc_min) / self.grid_size).to(torch.int)
            assert means3D_int_i.min() >= 0 and means3D_int_i[:, 0].max() < self.H and means3D_int_i[
                :, 1].max() < self.W and means3D_int_i[:, 2].max() < self.D

            radii_i = torch.ceil(scales_i.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
            assert radii_i.min() >= 1

            print(f"Radii: {radii_i}")
            print(f"Length: {len(radii_i)}")
            print(f"Radii max: {radii_i.max()}, min: {radii_i.min()}")

            cov3D_flat_i = cov3D_i.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

            # Invoke C++/CUDA rasterization routine for the single sample
            logits_i = _LocalAggregate.apply(
                pts_i,
                points_int_i,
                means3D_i,
                means3D_int_i,
                opacities_i,
                semantics_i,
                radii_i,
                cov3D_flat_i,
                self.H, self.W, self.D
            )
            output_logits.append(logits_i)

        # Stack the results from each sample back into a single batch tensor
        final_logits = torch.stack(output_logits, dim=0)
        # --- End of new batch handling logic ---

        if not self.inv_softmax:
            return final_logits  # n, c -> now (b, n, c) after cat
        else:
            assert False, "Inverse softmax not implemented for batched processing"