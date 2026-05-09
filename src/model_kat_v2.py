import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import config

# KATv2-specific constants (not in config.py — v2 only)
KAT2_NUM_AGENTS = 16
KAT2_NUM_HEADS  = 4
KAT2_INIT_STD   = 0.02
KAT2_BLOCKS_IDX = 4   # EfficientNet-B0 blocks[4] → (B, 112, 14, 14)
KAT2_FEAT_CH    = 112  # channels at blocks[4] output


class KATModelV2(nn.Module):
    """
    KATv2 — Kernel-guided Agent Transformer, version 2.

    Differences from KATv1 (model_kat.py):
    - Feature map: blocks[4] via forward hook → (B, 112, 14, 14) instead of
      forward_features → (B, 1280, 7, 7). Spatial tokens: 196 vs 49.
    - Agents: 16 instead of 8.
    - Attention: 4-head cross-attention (manual torch.matmul, MPS-safe)
      instead of single-head.

    Architecture:
        EfficientNet-B0 (frozen) → hook on blocks[4] → (B, 112, 14, 14)
        → 1x1 Conv projection   → (B, 256, 14, 14)
        → flatten               → (B, 196, 256)
        → 4-head cross-attn     → (B, 16, 256)   [16 agents]
        → LayerNorm → Flatten   → (B, 4096)
        → MLP                   → (B, NUM_CLASSES)
    """

    def __init__(self,
                 num_classes=config.NUM_CLASSES,
                 backbone_name=config.BACKBONE,
                 pretrained=config.PRETRAINED,
                 num_agents=KAT2_NUM_AGENTS,
                 num_heads=KAT2_NUM_HEADS,
                 agent_dim=config.KAT_AGENT_DIM,
                 mlp_hidden=config.KAT_MLP_HIDDEN,
                 dropout=config.KAT_DROPOUT):
        super().__init__()

        assert agent_dim % num_heads == 0, \
            f"agent_dim ({agent_dim}) must be divisible by num_heads ({num_heads})"

        self.num_agents = num_agents
        self.num_heads  = num_heads
        self.agent_dim  = agent_dim
        self.head_dim   = agent_dim // num_heads

        # Backbone — frozen by default
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Hook to capture blocks[KAT2_BLOCKS_IDX] output
        self._feat_cache = {}
        self.backbone.blocks[KAT2_BLOCKS_IDX].register_forward_hook(
            lambda m, inp, out: self._feat_cache.__setitem__('x', out)
        )

        # 1x1 projection: 112 → agent_dim
        self.proj = nn.Conv2d(KAT2_FEAT_CH, agent_dim, kernel_size=1)

        # Learnable agent queries (num_agents, agent_dim)
        self.agent_queries = nn.Parameter(
            torch.randn(num_agents, agent_dim) * KAT2_INIT_STD
        )

        # Output projection after multi-head concat
        self.out_proj = nn.Linear(agent_dim, agent_dim)

        # Per-agent LayerNorm
        self.agent_norm = nn.LayerNorm(agent_dim)

        # MLP classifier
        flattened_dim = num_agents * agent_dim   # 16 * 256 = 4096
        h1 = mlp_hidden[0] if len(mlp_hidden) > 0 else max(flattened_dim // 8, 128)
        h2 = mlp_hidden[1] if len(mlp_hidden) > 1 else max(h1 // 2, 128)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.8),
            nn.Linear(h2, num_classes),
        )

    def _multihead_attention(self, Q, K, V):
        """
        Manual 4-head scaled dot-product cross-attention. MPS-safe (torch.matmul only).

        Args:
            Q: (B, num_agents, agent_dim)
            K: (B, num_tokens, agent_dim)
            V: (B, num_tokens, agent_dim)

        Returns:
            out:  (B, num_agents, agent_dim)
            attn: (B, num_heads, num_agents, num_tokens)  — for visualisation
        """
        B, N, D = Q.shape
        _, K_len, _ = K.shape
        H, Dh = self.num_heads, self.head_dim

        # Split into heads: (B, seq, H, Dh) → (B, H, seq, Dh)
        Q = Q.reshape(B, N,     H, Dh).permute(0, 2, 1, 3)   # (B, H, N, Dh)
        K = K.reshape(B, K_len, H, Dh).permute(0, 2, 1, 3)   # (B, H, K_len, Dh)
        V = V.reshape(B, K_len, H, Dh).permute(0, 2, 1, 3)   # (B, H, K_len, Dh)

        # Scaled dot-product attention per head
        scale  = math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, N, K_len)
        attn   = F.softmax(scores, dim=-1)                      # (B, H, N, K_len)

        # Weighted sum
        out = torch.matmul(attn, V)          # (B, H, N, Dh)
        out = out.permute(0, 2, 1, 3)        # (B, N, H, Dh)
        out = out.reshape(B, N, D)           # (B, N, agent_dim)  — heads concatenated

        return out, attn

    def forward(self, x, return_attentions=False):
        """
        Args:
            x: (B, 3, H, W)
            return_attentions: if True, also return attn maps (B, num_heads, num_agents, 14, 14)

        Returns:
            logits (B, NUM_CLASSES)  or  (logits, attn_maps) if return_attentions
        """
        B = x.shape[0]

        # Clear cache before each forward pass
        self._feat_cache.clear()

        # Trigger hook; we only need blocks[4] output, not the final features
        self.backbone.forward_features(x)
        feats = self._feat_cache['x']          # (B, 112, 14, 14)

        if feats.dim() != 4:
            raise RuntimeError(f"Expected 4D feature map from blocks[{KAT2_BLOCKS_IDX}], got {feats.shape}")

        _, _, Hf, Wf = feats.shape

        # Project to agent_dim
        proj = self.proj(feats)                # (B, 256, 14, 14)

        # Flatten spatial tokens
        K_len = Hf * Wf                        # 196
        tokens = proj.view(B, self.agent_dim, K_len).permute(0, 2, 1)  # (B, 196, 256)

        # Expand agent queries to batch
        Q = self.agent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 16, 256)

        # 4-head cross-attention
        agent_out, attn = self._multihead_attention(Q, tokens, tokens)  # (B, 16, 256)

        # Output projection + norm
        agent_out = self.out_proj(agent_out)   # (B, 16, 256)
        agent_out = self.agent_norm(agent_out)

        # Flatten and classify
        flat   = agent_out.view(B, -1)         # (B, 4096)
        logits = self.classifier(flat)

        if return_attentions:
            # attn: (B, num_heads, num_agents, K_len) → (B, num_heads, num_agents, Hf, Wf)
            attn_maps = attn.view(B, self.num_heads, self.num_agents, Hf, Wf)
            return logits, agent_out, attn_maps

        return logits, agent_out


def build_kat_v2(**kwargs):
    return KATModelV2(**kwargs)


if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model  = KATModelV2().to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    x      = torch.zeros(2, 3, 224, 224, device=device)
    logits, agent_out, attn = model(x, return_attentions=True)
    print(f"logits shape:     {logits.shape}")
    print(f"agent_out shape:  {agent_out.shape}")
    print(f"attn_maps shape:  {attn.shape}")
