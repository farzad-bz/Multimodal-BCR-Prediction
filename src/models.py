import torch
import torch.nn as nn
from transformers import AutoModel

class SurvivalModelMM(nn.Module):

    """
    Modular multimodal survival model.

    - Each modality has its own Linear -> LayerNorm -> ReLU -> Dropout block
    that maps from its raw input dim to a shared embedding dim d_emb.
    - Any subset of modalities can be provided at forward().
    - Embeddings of all provided modalities are averaged, then passed to
    a small MLP to predict risk.

    Example modalities:
        modalities = {
            "clinical": 6,     # len(features)
            "t2": 768,         # M3D-CLIP T2 emb
            "hbv": 768,        # M3D-CLIP HBV emb
            "adc": 768,        # M3D-CLIP ADC emb
        }
    """
    def __init__(
        self,
        modalities: dict,   # e.g. {"clinical": 6, "t2": 768, "hbv": 768, "adc": 768}
        d_emb: int = 16,          # embedding dim for EACH modality
        dropout: float = 0.2,
    ):
        super().__init__()

        self.modalities = modalities
        self.d_emb = d_emb

        # One projection + norm per modality
        self.proj = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for name, in_dim in modalities.items():
            self.proj[name] = nn.Linear(in_dim, d_emb)
            self.norm[name] = nn.LayerNorm(d_emb)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Fusion head operates on a single fused embedding of size d_emb
        self.fusion = nn.Sequential(
            nn.Linear(d_emb, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, inputs: dict):
        """
        inputs: dict mapping modality_name -> tensor
            - Each tensor should be of shape (B, modalities[name]).
            - All provided modalities must have the same batch size B.

        Examples:
            risk = model({"clinical": x_clin})
            risk = model({"clinical": x_clin, "t2": t2_emb})
            risk = model({"clinical": x_clin, "t2": t2_emb, "hbv": hbv_emb})
            risk = model({"t2": t2_emb, "adc": adc_emb})
        """
        assert len(inputs) > 0, "At least one modality must be provided"

        embs = []
        for name, x in inputs.items():
            if name not in self.proj:
                raise ValueError(f"Unknown modality '{name}'. "
                                f"Known modalities: {list(self.proj.keys())}")
            z = self.proj[name](x)          # (B, d_emb)
            z = self.norm[name](z)          # (B, d_emb)
            z = self.act(z)
            z = self.dropout(z)
            embs.append(z)

        # fuse: mean over available modalities -> (B, d_emb)
        if len(embs) == 1:
            fused = embs[0]
        else:
            fused = torch.stack(embs, dim=0).mean(dim=0)

        risk = self.fusion(fused).squeeze(-1)  # (B,)
        return risk
    
    
def get_image_encoder(cfg, device):
    M3D_model = AutoModel.from_pretrained(
        cfg.image_encoder.pretrained_path,
        trust_remote_code=True
    )
    M3D_model = M3D_model.to(device=device)
    M3D_model.requires_grad_(False)
    M3D_model.eval()
    return M3D_model