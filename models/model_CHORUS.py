from __future__ import absolute_import, division, print_function
import logging
import math
from os.path import join as pjoin


logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.nn import functional as F

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize


class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, conch_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        self.tokenizer = get_tokenizer()

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(self.tokenizer, [ctx_init])
            with torch.no_grad():
                embedding = conch_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = tokenize(self.tokenizer, prompts)
        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = [len(self.tokenizer.encode(name, max_length=127, truncation=True))
                          for name in classnames]
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        else:
            raise NotImplementedError
        return prompts




# =====================================================================
# NEW: CHORUS - uncertainty-aware multi-encoder fusion
# =====================================================================

from cca.alignment import FrozenAlignment
from cca.uncertainty_attention import (
    reweight_features,
    rho_weighted_normalize,
    per_patch_reliability_bias,
)


class CHORUS(nn.Module):
    """
    Uncertainty-aware multi-encoder fusion via frozen GCCA, with
    optional per-stage uncertainty modulation.

    Forward signature mirrors upstream models:
        x_s_list : list of K tensors (1, N, d_k) - unused (API parity)
        x_l_list : list of K tensors (1, N, d_k) - high-res features
        label    : (1,)
        returns Y_prob, Y_hat, loss
    """

    def __init__(self, config, num_classes=3):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.window_size = config.window_size
        self.sim_threshold = config.sim_threshold
        self.L = config.input_size
        self.D = 512
        self.L_max = config.max_context_length

        # CHORUS: variant + learnable scalars
        self.variant = getattr(config, "chorus_variant", "none")
        self.alpha = nn.Parameter(torch.tensor(float(getattr(config, "chorus_alpha_init", 1.0))))
        self.beta = nn.Parameter(torch.tensor(float(getattr(config, "chorus_beta_init", 1.0))))
        self.tau = float(getattr(config, "chorus_tau", 0.3))

        # Per-stage disable flags (for ablation)
        self.disable_prestage = bool(getattr(config, "disable_prestage", False))
        self.disable_stage1_rho = bool(getattr(config, "disable_stage1_rho", False))
        self.disable_stage2_sigma = bool(getattr(config, "disable_stage2_sigma", False))
        self.disable_stage3_rho = bool(getattr(config, "disable_stage3_rho", False))
        self.disable_stage4_qk = bool(getattr(config, "disable_stage4_qk", False))
        self.disable_stage4_v = bool(getattr(config, "disable_stage4_v", False))
        self.disable_stage4_sigma = bool(getattr(config, "disable_stage4_sigma", False))

        # Visual alignment (REQUIRED)
        cca_visual_path = getattr(config, "cca_visual_path", None)
        if cca_visual_path is None:
            raise ValueError("CHORUS requires --cca_visual_path")
        self.visual_align = FrozenAlignment(cca_visual_path)
        assert self.visual_align.latent_dim == self.D, \
            f"CCA latent_dim ({self.visual_align.latent_dim}) != model D ({self.D})"
        self.K_v = self.visual_align.K

        # Text alignment (OPTIONAL)
        cca_text_path = getattr(config, "cca_text_path", None)
        if cca_text_path is not None:
            self.text_align = FrozenAlignment(cca_text_path)
            self.K_t = self.text_align.K
        else:
            self.text_align = None
            self.K_t = 1
            self.register_buffer("rho_t_default", torch.ones(self.D))

        # CONCH backbone (text encoder)
        conch_model_cfg = 'conch_ViT-B-16'
        conch_checkpoint_path = 'ckpts/conch.pth'
        conch_model, _ = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)
        _ = conch_model.eval()

        # Freeze backbone
        for param in conch_model.parameters():
            param.requires_grad = False

        self.feature_dim = conch_model.text.text_projection.shape[1]
        self.prompt_learner = PromptLearner(config.text_prompt, conch_model.float())
        self.text_encoder = TextEncoder(conch_model.float())

        self.feature_encoder = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.LayerNorm(self.D),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        num_heads = 8
        self.head_dim = self.feature_dim // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.o_proj = nn.Linear(self.feature_dim, self.feature_dim)

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    # helpers
    def _get_rho_v(self):
        return self.visual_align.rho

    def _get_rho_t(self):
        return self.text_align.rho if self.text_align is not None else self.rho_t_default

    def _fuse(self, x_l_list):
        z_list = [x.squeeze(0).float() for x in x_l_list]
        mu, sigma_sq, _ = self.visual_align(z_list)
        return mu, sigma_sq

    # Stage 1
    def compute_patch_similarity_chorus(self, x, sigma_sq, rho_v, window_size):
        N, D = x.shape
        eff = "none" if self.disable_stage1_rho else self.variant
        x_norm = rho_weighted_normalize(x, rho_v, eff, tau=self.tau)
        similarities, selected_indices = [], []
        for i in range(0, N, window_size):
            window = x_norm[i:i + window_size]
            if len(window) < 2:
                selected_indices.append(torch.arange(i, min(i + window_size, N), device=x.device))
                continue
            window_sim = torch.mm(window, window.t())
            if window_sim.numel() > 1:
                threshold = window_sim.mean() + window_sim.std(unbiased=False)
            else:
                threshold = window_sim.mean()
            redundant = window_sim.mean(1) > threshold
            keep_indices = torch.where(~redundant)[0] + i
            if len(keep_indices) == 0:
                keep_indices = torch.tensor([i], device=x.device)
            selected_indices.append(keep_indices)
            similarities.append(window_sim)
        if not selected_indices:
            return [], torch.arange(N, device=x.device)
        return similarities, torch.cat(selected_indices)

    # Stage 2
    def adaptive_token_selection_chorus(self, features, sigma_sq, rho_v, text_features):
        N, D = features.shape
        _, indices = self.compute_patch_similarity_chorus(features, sigma_sq, rho_v, self.window_size)

        eff_pre = "none" if self.disable_prestage else self.variant
        feats_for_score = reweight_features(
            features, sigma_sq, rho_v,
            variant=eff_pre, alpha=self.alpha.item(), tau=self.tau,
        )
        if feats_for_score.shape[-1] != text_features.shape[-1]:
            projection = nn.Linear(feats_for_score.shape[-1], text_features.shape[-1],
                                   device=feats_for_score.device)
            feats_for_score = projection(feats_for_score)
        text_relevance = torch.matmul(feats_for_score, text_features.T).mean(-1)

        if (self.variant in ("v3_local", "v4_hybrid")
                and not self.disable_stage2_sigma):
            patch_penalty = self.beta * sigma_sq.sum(dim=-1)
            text_relevance = text_relevance - patch_penalty

        importance_mask = torch.zeros(N, device=features.device, dtype=text_relevance.dtype)
        importance_mask[indices] = text_relevance[indices]
        num_tokens = min(self.L_max, N)
        _, selected_indices = torch.topk(importance_mask, num_tokens)
        selected_indices, _ = torch.sort(selected_indices)
        selected_features = features[selected_indices]
        selected_sigma_sq = sigma_sq[selected_indices]
        return selected_features, selected_sigma_sq, selected_indices

    # Stage 3
    def spatial_token_compression_chorus(self, features, sigma_sq, rho_v):
        N, D = features.shape
        eff = "none" if self.disable_stage3_rho else self.variant
        chunk_size = 8
        compressed_feats, compressed_sigma = [], []
        for i in range(0, N, chunk_size):
            chunk = features[i:i + chunk_size]
            chunk_sigma = sigma_sq[i:i + chunk_size]
            if len(chunk) == 1:
                compressed_feats.append(chunk)
                compressed_sigma.append(chunk_sigma)
                continue
            chunk_norm = rho_weighted_normalize(chunk, rho_v, eff, tau=self.tau)
            sim = F.cosine_similarity(chunk_norm[:-1], chunk_norm[1:], dim=-1)
            keep_mask = sim < self.sim_threshold
            kept_chunk = torch.cat([chunk[:1], chunk[1:][keep_mask]])
            kept_chunk_sigma = torch.cat([chunk_sigma[:1], chunk_sigma[1:][keep_mask]])
            compressed_feats.append(kept_chunk)
            compressed_sigma.append(kept_chunk_sigma)
        feats = torch.cat(compressed_feats)
        sig = torch.cat(compressed_sigma)
        if len(feats) > self.L_max:
            feats = feats[:self.L_max]
            sig = sig[:self.L_max]
        return feats, sig

    # Stage 4
    def cross_attention_chorus(self, queries, keys, values, sigma_sq_v,
                              rho_v, rho_t, attention_mask=None):
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        if not self.disable_stage4_qk:
            if self.variant in ("v2_attn", "v4_hybrid"):
                sqrt_rho_v = torch.sqrt(torch.clamp(rho_v, min=0.0))
                k = k * sqrt_rho_v
                if rho_t is not None:
                    sqrt_rho_t = torch.sqrt(torch.clamp(rho_t, min=0.0))
                    q = q * sqrt_rho_t
            elif self.variant == "v10_temp":
                sqrt_rho_v = torch.sqrt(torch.clamp(rho_v, min=0.0))
                q = q * sqrt_rho_v
                k = k * sqrt_rho_v
            elif self.variant == "v9_inverse":
                sqrt_inv = torch.sqrt(torch.clamp(1.0 - rho_v, min=0.0))
                k = k * sqrt_inv

        if not self.disable_stage4_v and self.variant == "v4_hybrid":
            v = v * rho_v

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale

        if (self.variant in ("v3_local", "v4_hybrid")
                and not self.disable_stage4_sigma
                and sigma_sq_v is not None):
            patch_bias = per_patch_reliability_bias(sigma_sq_v, self.beta)
            scores = scores + patch_bias.unsqueeze(1).unsqueeze(2)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.feature_dim)
        return self.o_proj(out)

    # Forward
    def forward(self, x_s_list, x_l_list, label):
        # PHASE 2: align + fuse
        mu, sigma_sq = self._fuse(x_l_list)
        rho_v = self._get_rho_v().to(mu.device).to(mu.dtype)
        rho_t = self._get_rho_t().to(mu.device).to(mu.dtype)

        # Pre-stage feature reweighting
        eff_pre = "none" if self.disable_prestage else self.variant
        mu_for_encoder = reweight_features(
            mu, sigma_sq, rho_v,
            variant=eff_pre, alpha=self.alpha.item(), tau=self.tau,
        )

        # Text features
        prompts = self.prompt_learner()
        text_features = self.text_encoder(
            prompts, self.prompt_learner.tokenized_prompts
        )[self.num_classes:]

        # MLP encode
        features = self.feature_encoder(mu_for_encoder)

        # Stages 1+2
        selected_features, selected_sigma, _ = self.adaptive_token_selection_chorus(
            features, sigma_sq, rho_v, text_features
        )

        # Stage 3
        compressed_features, compressed_sigma = self.spatial_token_compression_chorus(
            selected_features, selected_sigma, rho_v
        )

        # Prepare for Stage 4
        compressed_features = compressed_features.unsqueeze(0)
        compressed_sigma = compressed_sigma.unsqueeze(0)
        text_features_b = text_features.unsqueeze(0)

        # Stage 4
        attended = self.cross_attention_chorus(
            queries=text_features_b,
            keys=compressed_features,
            values=compressed_features,
            sigma_sq_v=compressed_sigma,
            rho_v=rho_v,
            rho_t=rho_t,
        )

        final = attended.mean(1)
        logits = self.classifier(final)
        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        return Y_prob, Y_hat, loss
