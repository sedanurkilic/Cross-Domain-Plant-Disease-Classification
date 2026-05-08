import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import config

# Local KAT constants (kept inside model file as requested)
KAT_ATT_HEADS = 1
KAT_INIT_STD = 0.02


class KATModel(nn.Module):
    """
    Kernel-guided Agent Transformer (KAT) model.

    Architecture (high level):
    - EfficientNet-B0 backbone (timm, pretrained, num_classes=0) -> features (B, C, H, W)
    - 1x1 conv projection to D = KAT_AGENT_DIM -> (B, D, H, W)
    - N learnable agent queries (N = KAT_NUM_AGENTS) of dim D
    - Scaled-dot-product cross-attention: agents (queries) attend to spatial tokens (keys/values)
    - Agent outputs (B, N, D) flattened -> MLP -> class logits (B, NUM_CLASSES)

    The backbone parameters are frozen by default to follow the few-shot / transfer workflow.
    """

    def __init__(self,
                 num_classes=config.NUM_CLASSES,
                 backbone_name=config.BACKBONE,
                 pretrained=config.PRETRAINED,
                 num_agents=config.KAT_NUM_AGENTS,
                 agent_dim=config.KAT_AGENT_DIM,
                 mlp_hidden=config.KAT_MLP_HIDDEN,
                 dropout=config.KAT_DROPOUT):
        super().__init__()

        # Build backbone (same pattern as src/model.py)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.backbone_out_channels = getattr(self.backbone, 'num_features', None)
        if self.backbone_out_channels is None:
            raise RuntimeError("Backbone does not expose num_features attribute")

        # Freeze backbone parameters by default
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 1x1 projection from backbone channels -> agent_dim
        self.proj = nn.Conv2d(self.backbone_out_channels, agent_dim, kernel_size=1)

        # Learnable agent queries (N, D)
        self.num_agents = num_agents
        self.agent_dim = agent_dim
        self.agent_queries = nn.Parameter(torch.randn(num_agents, agent_dim) * KAT_INIT_STD)

        # Small layernorm on agent outputs (per-agent)
        self.agent_norm = nn.LayerNorm(agent_dim)

        # MLP classifier: flattened (N*D) -> mlp_hidden[0] -> mlp_hidden[1] -> num_classes
        flattened_dim = num_agents * agent_dim
        h1 = mlp_hidden[0] if len(mlp_hidden) > 0 else max(flattened_dim // 4, 128)
        h2 = mlp_hidden[1] if len(mlp_hidden) > 1 else max(h1 // 2, 128)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.8),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x, return_attentions: bool = False):
        """
        Forward pass.

        Args:
            x: input images (B, 3, H, W)
            return_attentions: if True, also return attention maps of shape (B, N, Hf, Wf)

        Returns:
            logits (B, NUM_CLASSES)  or (logits, attn_maps) if return_attentions
        """
        # Backbone features
        feats = self.backbone.forward_features(x)  # expected (B, C, Hf, Wf) or (B, C)

        # Validate backbone features shape.
        # We expect a spatial feature map (B, C, H, W) from forward_features.
        # If a 2D tensor is returned (B, C), that's likely due to a backbone
        # configuration or timm version that performs global pooling; fail fast
        # with an explicit, actionable error message so the user can fix it.
        if feats.dim() == 4:
            B, C, Hf, Wf = feats.shape
        elif feats.dim() == 2:
            raise RuntimeError(
                "Backbone forward_features returned 2D tensor (B, C). "
                "EfficientNet-B0 with timm should return (B, C, H, W). "
                "Check timm version or backbone config."
            )
        else:
            raise RuntimeError(f"Unexpected backbone feature shape: {feats.shape}")

        # Project to agent_dim
        proj = self.proj(feats)  # (B, D, Hf, Wf)
        D = proj.shape[1]

        # Flatten spatial tokens
        K = Hf * Wf
        proj_flat = proj.view(B, D, K).permute(0, 2, 1)  # (B, K, D)

        # Prepare queries (agents): (B, N, D)
        # agent_queries is (N, D) -> expand to batch
        Q = self.agent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # Scaled dot-product attention between Q and K
        # scores: (B, N, K)
        # use torch.matmul for MPS compatibility
        scores = torch.matmul(Q, proj_flat.transpose(1, 2))  # (B, N, K)
        scores = scores / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)  # attention over spatial tokens

        # Weighted sum over values (proj_flat is V)
        agent_out = torch.matmul(attn, proj_flat)  # (B, N, D)

        # Normalize per-agent vector
        agent_out = self.agent_norm(agent_out)

        # Flatten agents
        flat = agent_out.view(B, -1)  # (B, N*D)

        logits = self.classifier(flat)

        if return_attentions:
            attn_maps = attn.view(B, self.num_agents, Hf, Wf)
            return logits, attn_maps

        return logits


def build_kat_model(**kwargs):
    """Convenience builder for external scripts to match project pattern."""
    model = KATModel(**kwargs)
    return model


if __name__ == "__main__":
    # Small smoke instantiation (not executed automatically by CI) — kept for manual quick checks
    device = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = KATModel()
    model.to(device)
    print("KATModel created. Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))