"""
cca/uncertainty_attention.py
----------------------------
Stateless helpers for all CHORUS variants. NEW FILE.
"""

import torch
import torch.nn.functional as F


def reweight_features(mu, sigma_sq, rho, variant, alpha=1.0, tau=0.3):
    """Per-feature reweighting: mu_tilde = g(mu, sigma^2, rho)"""
    if variant == "none":
        return mu
    if variant == "v1_embed":
        return mu * rho
    if variant == "v5_precision":
        w = rho / (1.0 + alpha * sigma_sq)
        return mu * w
    if variant == "v6_threshold":
        return mu * (rho > tau).to(mu.dtype)
    if variant == "v9_inverse":
        return mu * (1.0 - rho)
    return mu


def rho_weighted_normalize(x, rho, variant, eps=1e-8, tau=0.3):
    """rho-weighted L2 normalization for cosine similarity."""
    if variant == "none":
        return F.normalize(x, p=2, dim=-1, eps=eps)
    if variant == "v9_inverse":
        weight = torch.sqrt(torch.clamp(1.0 - rho, min=0.0))
    elif variant == "v6_threshold":
        weight = (rho > tau).to(x.dtype)
    else:
        weight = torch.sqrt(torch.clamp(rho, min=0.0))
    xw = x * weight
    return F.normalize(xw, p=2, dim=-1, eps=eps)


def per_patch_reliability_bias(sigma_sq, beta):
    """Per-patch reliability bias: -beta * sum_d sigma^2_{i,d}"""
    return -beta * sigma_sq.sum(dim=-1)


def modulate_qkv_for_attention(q, k, v, rho_v, rho_t, variant):
    """Modulate Q, K, V before multi-head reshape (rho is along last dim)."""
    if variant in ("v2_attn", "v4_hybrid"):
        sqrt_rho_v = torch.sqrt(torch.clamp(rho_v, min=0.0))
        k = k * sqrt_rho_v
        if rho_t is not None:
            sqrt_rho_t = torch.sqrt(torch.clamp(rho_t, min=0.0))
            q = q * sqrt_rho_t
    elif variant == "v10_temp":
        sqrt_rho_v = torch.sqrt(torch.clamp(rho_v, min=0.0))
        q = q * sqrt_rho_v
        k = k * sqrt_rho_v
    elif variant == "v9_inverse":
        sqrt_inv = torch.sqrt(torch.clamp(1.0 - rho_v, min=0.0))
        k = k * sqrt_inv
    if variant == "v4_hybrid":
        v = v * rho_v
    return q, k, v
