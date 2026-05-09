"""
cca/alignment.py
----------------
FrozenAlignment: nn.Module wrapping CCA projections + means + rho as
non-trainable buffers. Used inside CHORUS.
NEW FILE.
"""

import numpy as np
import torch
import torch.nn as nn


class FrozenAlignment(nn.Module):
    """
    Frozen CCA alignment of K encoders into a shared d-dim space.
    forward(z_list) -> (mu, sigma_sq, rho)
        z_list : list of K tensors (..., d_k)  raw per-encoder features
        mu       : (..., d)  cross-encoder mean
        sigma_sq : (..., d)  per-element local uncertainty
        rho      : (d,)      global per-dim agreement (frozen)
    """

    def __init__(self, npz_path):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        W_list = list(data["W_list"])
        means = list(data["means"])
        rho = data["rho"]
        self.K = len(W_list)
        self.latent_dim = int(data["latent_dim"])
        if "encoders" in data.files:
            self.names = list(data["encoders"])
        elif "sources" in data.files:
            self.names = list(data["sources"])
        else:
            self.names = [f"enc_{k}" for k in range(self.K)]

        for k in range(self.K):
            self.register_buffer(f"W_{k}",
                                 torch.from_numpy(W_list[k].astype(np.float32)))
            self.register_buffer(f"mean_{k}",
                                 torch.from_numpy(means[k].astype(np.float32)))
        self.register_buffer("rho", torch.from_numpy(rho.astype(np.float32)))

    def forward(self, z_list):
        assert len(z_list) == self.K, \
            f"Got {len(z_list)} encoders, expected {self.K}"
        h_list = []
        for k in range(self.K):
            W = getattr(self, f"W_{k}")
            mu_k = getattr(self, f"mean_{k}")
            h_list.append((z_list[k] - mu_k) @ W)
        stacked = torch.stack(h_list, dim=0)
        mu = stacked.mean(dim=0)
        if self.K == 1:
            sigma_sq = torch.zeros_like(mu)
        else:
            sigma_sq = stacked.var(dim=0, unbiased=False)
        return mu, sigma_sq, self.rho

    def __repr__(self):
        return (f"FrozenAlignment(K={self.K}, d={self.latent_dim}, "
                f"sources={self.names}, rho_mean={self.rho.mean().item():.3f})")
